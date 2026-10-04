# hod_2d

A pure-NumPy Extended HOD mock galaxy catalog generator with 2D Assembly Bias (Concentration + Environment) and Jeans Kinematics, optimized for Simulation-Based Inference (SBI).

## Installation

```bash
pip install git+https://github.com/amirreza-kolahmal/hod_2d.git

QuickStart

import pandas as pd
import numpy as np
from fast_sbi_hod import precompute_halo_ranks, generate_extended_hod_mock

# Load catalog
df_halos = pd.DataFrame(np.load("rockstar_halos.npy"))

# Precompute assembly bias ranks ONCE
conc_ranks, env_ranks = precompute_halo_ranks(df_halos, box_size=400.0)

# Generate mock galaxy catalog
galaxy_mock = generate_extended_hod_mock(
    df=df_halos,
    conc_ranks=conc_ranks,
    env_ranks=env_ranks,
    hod_params={"logMmin": 12.8, "enable_conc_bias": True},
    apply_rsd=True
)
