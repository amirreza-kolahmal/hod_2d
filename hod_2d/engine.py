from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from scipy.integrate import quad
from astropy.cosmology import Planck15, FLRW
import astropy.constants as const
import astropy.units as u


# ==============================================================================
# 1. ASTROPY CONSTANTS & LOOKUP TABLES
# ==============================================================================
# G pulled from Astropy in (Mpc) * (km/s)^2 / M_sun
G_CONSTANT = const.G.to(u.Mpc * (u.km / u.s)**2 / u.Msun).value


def nfw_enclosed_mass(x: np.ndarray) -> np.ndarray:
    """Dimensionless NFW enclosed mass profile g(x) = ln(1+x) - x/(1+x)."""
    return np.log(1.0 + x) - (x / (1.0 + x))


def build_jeans_interpolator() -> Tuple[np.ndarray, np.ndarray]:
    """
    Precomputes dimensionless 1D velocity dispersion profile sigma_tilde(x)
    by numerically integrating the Jeans equation for isotropic NFW profiles.
    """
    x_grid = np.logspace(-4, 2.5, 1000)
    
    def jeans_integrand(v: float) -> float:
        return (np.log(1.0 + v) - (v / (1.0 + v))) / ((v**3) * (v + 1.0)**2)

    integrals = np.array([quad(jeans_integrand, x, np.inf)[0] for x in x_grid])
    dim_sigma = np.sqrt((x_grid * (1.0 + x_grid)**2) * integrals)
    
    dim_sigma = np.insert(dim_sigma, 0, 0.0)
    x_grid = np.insert(x_grid, 0, 0.0)
    
    return x_grid, dim_sigma


# Global Precomputed Lookup Tables (Built ONCE on import)
JEANS_X_GRID, JEANS_DIM_SIGMA = build_jeans_interpolator()
NFW_X_GRID = np.insert(np.logspace(-4, 2.5, 10000), 0, 0.0)
NFW_G_GRID = nfw_enclosed_mass(NFW_X_GRID)


# ==============================================================================
# 2. HELPER UTILITIES
# ==============================================================================
def calculate_rsd_factor(redshift: float, cosmo: FLRW = Planck15) -> float:
    """
    Computes velocity-to-displacement conversion factor (km/s to Mpc/h)
    for Redshift-Space Distortions (RSD).
    """
    h = cosmo.H0.value / 100.0
    H_z = cosmo.H(redshift).value
    return (H_z / h) / (1.0 + redshift)


def get_mass_conditioned_ranks(
    halo_masses: np.ndarray, 
    values: np.ndarray, 
    n_bins: int = 100
) -> np.ndarray:
    """
    Computes mass-conditioned percentile ranks [0.0 to 1.0] using equal-count 
    mass binning (Halotools-style).
    """
    ranks = np.zeros_like(values, dtype=float)
    mass_sort_idx = np.argsort(halo_masses)
    bin_chunks = np.array_split(mass_sort_idx, n_bins)
    
    for chunk in bin_chunks:
        n_in_bin = len(chunk)
        if n_in_bin > 1:
            bin_vals = values[chunk]
            raw_ranks = rankdata(bin_vals, method="average")
            ranks[chunk] = (raw_ranks - 0.5) / n_in_bin
        else:
            ranks[chunk] = 0.5
            
    return ranks


# ==============================================================================
# 3. SBI PRECOMPUTATION UTILITY
# ==============================================================================
def precompute_halo_ranks(
    df: Any, 
    box_size: float = 400.0, 
    r_env: float = 5.0, 
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precomputes mass-conditioned concentration and environment density ranks 
    ONCE before starting SBI inference loops. Auto-sanitizes units if needed.
    """
    # Auto-convert Astropy Tables, dicts, or NumPy records to Pandas
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # 1. Input Validation
    required_cols = ["mvir", "rvir", "rs", "x", "y", "z"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input catalog is missing required columns: {missing}")
        
    # 2. Auto-sanitize Rockstar units (kpc/h to Mpc/h)
    if df["rvir"].max() > 100.0:
        df["rvir"] = df["rvir"] / 1000.0
        df["rs"] = df["rs"] / 1000.0

    if "halo_nfw_conc" not in df.columns:
        df["halo_nfw_conc"] = df["rvir"] / df["rs"]

    # 3. Compute Ranks
    halo_masses = df["mvir"].to_numpy()
    halo_coords = np.mod(df[["x", "y", "z"]].to_numpy(), box_size)
    halo_coords = np.clip(halo_coords, 0.0, box_size - 1e-7)
    
    conc_ranks = get_mass_conditioned_ranks(halo_masses, df["halo_nfw_conc"].to_numpy(), n_bins=n_bins)
    
    tree = cKDTree(halo_coords, boxsize=box_size)
    neighbor_indices = tree.query_ball_point(halo_coords, r=r_env, workers=-1)
    env_mass_density = np.array([halo_masses[idx].sum() for idx in neighbor_indices])
    
    env_ranks = get_mass_conditioned_ranks(halo_masses, env_mass_density, n_bins=n_bins)
    
    return conc_ranks, env_ranks


# ==============================================================================
# DEFAULT HOD PARAMETERS
# ==============================================================================
DEFAULT_HOD_PARAMS = {
    "logMmin": 12.7,
    "sigma_logM": 0.35,
    "logM1": 13.8,
    "alpha": 1.0,
    "alpha_c": 0.0,      # Default: no velocity bias
    "alpha_s": 1.0,      # Default: perfect Jeans kinematics
    "c_gal_bias": 1.0,
    "A_cent": 0.0,       # Default: no concentration bias
    "A_sat": 0.0,
    "B_cent": 0.0,       # Default: no environment bias
    "B_sat": 0.0,
    "enable_conc_bias": False,
    "enable_env_bias": False
}


# ==============================================================================
# 4. EXTENDED HOD MOCK ENGINE
# ==============================================================================
def generate_extended_hod_mock(
    df: Any, 
    conc_ranks: np.ndarray, 
    env_ranks: np.ndarray, 
    hod_params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    apply_rsd: bool = True,
    rsd_factor: Optional[float] = None,
    redshift: float = 0.55,
    cosmo: FLRW = Planck15,
    box_size: float = 400.0
) -> pd.DataFrame:
    """
    Generates a full 3D mock galaxy catalog with 2D Assembly Bias (Concentration + Environment),
    Velocity Bias, Jeans Kinematics, and optional Redshift-Space Distortions (RSD).
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # 1. Smart Parameter Merging
    params = DEFAULT_HOD_PARAMS.copy()
    if hod_params is not None:
        unrecognized = [k for k in hod_params.keys() if k not in params]
        if unrecognized:
            import warnings
            warnings.warn(f"Unrecognized HOD parameters ignored: {unrecognized}")
        params.update(hod_params)
        
    if rng is None:
        rng = np.random.default_rng()

    if apply_rsd and (rsd_factor is None):
        rsd_factor = calculate_rsd_factor(redshift, cosmo)

    # 2. Extract Data
    halo_coords = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    halo_velocities = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float32)
    halo_masses = df["mvir"].to_numpy(dtype=np.float32)
    halo_vrms = df["vrms"].to_numpy(dtype=np.float32)
        
    c_flag = 1.0 if params["enable_conc_bias"] else 0.0
    e_flag = 1.0 if params["enable_env_bias"] else 0.0
    
    # --------------------------------------------------------------------------
    # A. CENTRAL GALAXIES OCCUPATION & KINEMATICS
    # --------------------------------------------------------------------------
    m_cut = 10**params["logMmin"]
    p_cent_base = 0.5 * sp.erfc(np.log10(m_cut / halo_masses) / params["sigma_logM"])
    
    raw_shift_cent = (c_flag * params["A_cent"] * (conc_ranks - 0.5) +
                      e_flag * params["B_cent"] * (env_ranks - 0.5))
    cent_shift = 4.0 * (p_cent_base * (1.0 - p_cent_base)) * raw_shift_cent
    p_cent_eff = np.clip(p_cent_base + cent_shift, 0.0, 1.0)
    
    rand_fs = rng.uniform(0.0, 1.0, len(halo_masses))
    has_central = p_cent_eff > rand_fs
    df["has_central"] = has_central.astype(int)
    
    sigma_1d = halo_vrms / np.sqrt(3.0)
    
    cent_x = halo_coords[has_central, 0] % box_size
    cent_y = halo_coords[has_central, 1] % box_size
    cent_z = halo_coords[has_central, 2] % box_size
    
    cent_vx = halo_velocities[has_central, 0]
    cent_vy = halo_velocities[has_central, 1]
    cent_vz = halo_velocities[has_central, 2] + rng.normal(0.0, params["alpha_c"] * sigma_1d[has_central])
    
    if apply_rsd:
        cent_z = (cent_z + cent_vz / rsd_factor) % box_size

    # --------------------------------------------------------------------------
    # B. SATELLITE GALAXIES OCCUPATION
    # --------------------------------------------------------------------------
    m_1 = 10**params["logM1"]
    mass_diff = np.maximum(0.0, halo_masses - m_cut)
    mean_n_sat_base = (mass_diff / m_1) ** params["alpha"]
    
    sat_shift = (c_flag * params["A_sat"] * (conc_ranks - 0.5) +
                 e_flag * params["B_sat"] * (env_ranks - 0.5))
    mean_n_sat_eff = mean_n_sat_base * np.maximum(0.0, 1.0 + sat_shift)
    
    n_sat = rng.poisson(mean_n_sat_eff) * has_central.astype(int)
    df["n_sat"] = n_sat
    total_sat = np.sum(n_sat)
    
    if total_sat == 0:
        return pd.DataFrame({
            "x": cent_x, "y": cent_y, "z": cent_z,
            "vx": cent_vx, "vy": cent_vy, "vz": cent_vz,
            "is_central": 1
        })

    # --------------------------------------------------------------------------
    # C. SATELLITE POSITIONS
    # --------------------------------------------------------------------------
    sat_parent_coords = np.repeat(halo_coords, n_sat, axis=0)
    sat_parent_velocities = np.repeat(halo_velocities, n_sat, axis=0)
    sat_rs = np.repeat(df["rs"].to_numpy(), n_sat, axis=0)
    sat_rvir = np.repeat(df["rvir"].to_numpy(), n_sat, axis=0)
    sat_mvir = np.repeat(df["mvir"].to_numpy(), n_sat, axis=0)
    sat_c = np.repeat(df["halo_nfw_conc"].to_numpy(), n_sat, axis=0) * params["c_gal_bias"]
    
    u = rng.uniform(0.0, 1.0, total_sat)
    target_g = u * nfw_enclosed_mass(sat_c)
    sat_x = np.interp(target_g, NFW_G_GRID, NFW_X_GRID)
    sat_radii = sat_x * sat_rs
    
    phi = rng.uniform(0.0, 2.0 * np.pi, total_sat)
    costheta = rng.uniform(-1.0, 1.0, total_sat)
    sintheta = np.sqrt(1.0 - costheta**2)
    
    spatial_offset = np.column_stack([
        sat_radii * sintheta * np.cos(phi),
        sat_radii * sintheta * np.sin(phi),
        sat_radii * costheta
    ])
    
    sat_real_coords = (sat_parent_coords + spatial_offset) % box_size
    sat_z = sat_real_coords[:, 2]

    # --------------------------------------------------------------------------
    # D. SATELLITE KINEMATICS
    # --------------------------------------------------------------------------
    sat_dim_sigma = np.interp(sat_x, JEANS_X_GRID, JEANS_DIM_SIGMA)
    sat_vvir = np.sqrt((G_CONSTANT * sat_mvir) / sat_rvir)
    g_c = nfw_enclosed_mass(sat_c)
    amp_factor = np.sqrt(sat_c / g_c)
    
    local_sigma = sat_vvir * amp_factor * sat_dim_sigma * params["alpha_s"]
    
    sat_vx = sat_parent_velocities[:, 0] + rng.normal(0.0, local_sigma)
    sat_vy = sat_parent_velocities[:, 1] + rng.normal(0.0, local_sigma)
    sat_vz = sat_parent_velocities[:, 2] + rng.normal(0.0, local_sigma)
    
    if apply_rsd:
        sat_z = (sat_z + sat_vz / rsd_factor) % box_size
    
    # --------------------------------------------------------------------------
    # E. UNIFIED CATALOG
    # --------------------------------------------------------------------------
    df_cent = pd.DataFrame({
        "x": cent_x, "y": cent_y, "z": cent_z,
        "vx": cent_vx, "vy": cent_vy, "vz": cent_vz,
        "is_central": 1
    })
    
    df_sat = pd.DataFrame({
        "x": sat_real_coords[:, 0], "y": sat_real_coords[:, 1], "z": sat_z,
        "vx": sat_vx, "vy": sat_vy, "vz": sat_vz,
        "is_central": 0
    })
    
    return pd.concat([df_cent, df_sat], ignore_index=True)
