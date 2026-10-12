from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional, Union
import time
import numpy as np
import pandas as pd
from scipy.stats import qmc
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
from hod_2d import generate_extended_hod_mock, precompute_halo_ranks, calculate_rsd_factor

def sample_hod_priors(hod_priors: Dict[str, Tuple[float, float]],
                      n_samples: int = 30,
                      method: str = 'latin_hypercube',
                      seed: Optional[int] = None) -> List[Dict[str, Any]]:
    
    param_names = list(hod_priors.keys())
    n_dim = len(param_names)
    l_bounds = [bounds[0] for bounds in hod_priors.values()]
    u_bounds = [bounds[1] for bounds in hod_priors.values()]
    
    if method == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
        unit_samples = sampler.random(n=n_samples)
    
    elif method == "uniform":
        rng = np.random.default_rng(seed)
        unit_samples = rng.uniform(0.0, 1.0, size=(n_samples, n_dim))
        
    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    
    scaled_samples = qmc.scale(unit_samples, l_bounds, u_bounds)
    
    samples = []
    for row in scaled_samples:
        
        p_dict = dict(zip(param_names, row))
        
        p_dict["enable_conc_bias"] = True
        p_dict["enable_env_bias"] = True
        
        samples.append(p_dict)
    
    return samples
    

def generate_mocks_for_snapshot(
    df: pd.DataFrame,
    conc_ranks: np.ndarray,
    env_ranks: np.ndarray,
    hod_params_dicts: List[Dict[str, Any]],
    apply_rsd: bool = True,
    rsd_factor: Optional[float] = None,
    cosmo: Optional[FlatLambdaCDM] = None,
    box_size: float = 1000.0,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    
    
    snapshot_mocks = []
    for param_sample in hod_params_dicts:
       mock = generate_extended_hod_mock(df=df,
                                   conc_ranks=conc_ranks,
                                   env_ranks=env_ranks,
                                   hod_params=param_sample,
                                   apply_rsd=apply_rsd,
                                   rsd_factor=rsd_factor,
                                   cosmo=cosmo,
                                   box_size=box_size,
                                   rng=rng)
       snapshot_mocks.append((mock, param_sample))
    
    return snapshot_mocks

def process_single_realization(
    sim_id: int,
    catalog_dir: Union[str, Path],
    output_dir: Union[str, Path],
    all_params: np.ndarray,
    hod_priors: Dict[str, Tuple[float, float]],
    n_mocks_per_sim: int = 30,
    sampling_method: str = "latin_hypercube",
    box_size: float = 1000.0,
    redshift: float = 0.5,
    apply_rsd: bool = True,
    bin_method: str = "equal_count",
    output_format: str = "parquet",
    skip_existing: bool = True,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:

    cat_dir = Path(catalog_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = cat_dir / f"out_3_pid_{sim_id}.npy"
    if not file_path.exists():
        return []

    # 1. Setup Cosmology
    om, ob, h, ns, s8 = all_params[sim_id]
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=om, Ob0=ob)
    rsd_factor = calculate_rsd_factor(redshift=redshift, cosmo=cosmo)
    
    # 2. Load and Precompute ONCE
    df = pd.DataFrame(np.load(file_path))
    conc_ranks, env_ranks = precompute_halo_ranks(
        df=df, box_size=box_size, bin_method=bin_method
    )

    # 3. Generate Parameters & Mocks
    hod_params_dicts = sample_hod_priors(
        hod_priors=hod_priors, n_samples=n_mocks_per_sim, method=sampling_method, seed=seed
    )
    
    rng = np.random.default_rng(seed)
    snapshot_mocks = generate_mocks_for_snapshot(
        df=df, conc_ranks=conc_ranks, env_ranks=env_ranks,
        hod_params_dicts=hod_params_dicts, apply_rsd=apply_rsd,
        rsd_factor=rsd_factor, cosmo=cosmo, box_size=box_size, rng=rng
    )

    
    metadata_rows = []
    
    for mock_idx, (mock_df, param_sample) in enumerate(snapshot_mocks):
        filename = f"mock_lh_{sim_id:04d}_hod_{mock_idx:02d}.{output_format}"
        target_path = out_dir / filename

        if not (skip_existing and target_path.exists()):
            if output_format == "csv":
                mock_df.to_csv(target_path, index=False)
            elif output_format == "parquet":
                mock_df.to_parquet(target_path, index=False)
            elif output_format == "npy":
                np.save(target_path, mock_df.to_records(index=False))

        # Build the exact row for this mock for the Neural Network CSV
        row = param_sample.copy()
        row["sim_id"] = sim_id
        row["mock_idx"] = mock_idx
        row["filename"] = filename
        row.update({"om": om, "ob": ob, "h": h, "ns": ns, "s8": s8}) # True cosmology
        metadata_rows.append(row)

    # 5. Prevent memory leaks by explicitly deleting massive arrays
    del df
    del conc_ranks
    del env_ranks
    del snapshot_mocks

    return metadata_rows
    

def run_sbi_pipeline(
    sim_ids: List[int],
    catalog_dir: Union[str, Path],
    output_dir: Union[str, Path],
    hod_priors: Dict[str, Tuple[float, float]],
    params_url: str = "https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/latin_hypercube/latin_hypercube_params.txt",
    metadata_filename: str = "sbi_metadata_master.csv",
    n_mocks_per_sim: int = 30,
    sampling_method: str = "latin_hypercube",
    box_size: float = 1000.0,
    redshift: float = 0.5,
    apply_rsd: bool = True,
    output_format: str = "parquet",
    skip_existing: bool = True,
    seed: Optional[int] = 42
):
    
    all_params = np.loadtxt(params_url)
    master_metadata = []
    t_start = time.perf_counter()
    
    # 2. Loop through simulations with a progress bar
    for sim_id in tqdm(sim_ids, desc="Processing Realizations"):
        
        metadata_rows = process_single_realization(
            sim_id=sim_id,
            catalog_dir=catalog_dir,
            output_dir=output_dir,
            all_params=all_params,
            hod_priors=hod_priors,
            n_mocks_per_sim=n_mocks_per_sim,
            sampling_method=sampling_method,
            box_size=box_size,
            redshift=redshift,
            apply_rsd=apply_rsd,
            bin_method="equal_count",
            output_format=output_format,
            skip_existing=skip_existing,
            seed=seed
            )
        master_metadata.append(metadata_rows)
    
    print("\nExporting master metadata CSV...")
    df_meta = pd.DataFrame(master_metadata)
    df_meta.to_csv(metadata_filename, index=False)
    
    t_end = time.perf_counter()
    total_mocks = len(master_metadata)
    print("-" * 60)
    print(f"✓ Pipeline Complete in {t_end - t_start:.2f} seconds.")
    print(f"✓ Generated {total_mocks} total mocks across {len(sim_ids)} simulations.")
    print(f"✓ Metadata saved to: {metadata_filename}")



























