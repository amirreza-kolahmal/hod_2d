from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm



def compute_knn_cdfs(
    coords: np.ndarray,
    box_size: float,
    k_values: List[int],
    r_bins: np.ndarray,
    n_random_points: int = 100000,
    seed: Optional[int] = None
) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    query_points = rng.uniform(0.0, box_size, (n_random_points, 3))
    
    tree = cKDTree(coords, boxsize=box_size)
    distances, _ = tree.query(query_points, k=k_values)
    
    cdf_list = []
    for i in range(len(k_values)):
        if len(k_values) > 1:
            dist_k = distances[:, i]
        else:
            dist_k = distances
            
        counts = np.searchsorted(np.sort(dist_k), r_bins)
        cdf = counts / n_random_points
        cdf_list.append(cdf)
    
    return np.concatenate(cdf_list)


def process_single_mock(
    mock_file_path: Union[str, Path],
    box_size: float,
    k_values: List[int],
    r_bins: np.ndarray,
    n_random_points: int = 100000,
    seed: Optional[int] = None
) -> np.ndarray:
    
    expected_length = len(k_values) * len(r_bins)
    fallback_array = np.full(expected_length, np.nan)
    
    try:
        file_path = Path(mock_file_path)

        if not file_path.exists():
            print(f"Warning: File {file_path} not found.")
            return fallback_array
    
        df = pd.read_parquet(mock_file_path)
        coords = df[["x", "y", "z"]].to_numpy()
        
        features = compute_knn_cdfs(
            coords=coords,
            box_size=box_size,
            k_values=k_values,
            r_bins=r_bins,
            n_random_points=n_random_points,
            seed=seed)
        
        return features
    
    except Exception as e:
        
        print(f"Error processing {mock_file_path}: {e}")
        return fallback_array
        
        
def run_knn_pipeline(
    metadata_csv_path: Union[str, Path],
    mock_dir: Union[str, Path],
    output_feature_path: Union[str, Path],
    k_values: List[int] = [1, 2, 5],
    r_bins: Optional[np.ndarray] = None,
    box_size: float = 1000.0,
    n_random_points: int = 100000,
    n_workers: int = 4,
    limit: Optional[int] = None
):
    
    if r_bins is None:
        r_bins = np.linspace(0.1, 30.0, 50)
        
    mock_path = Path(mock_dir)
    df_meta = pd.read_csv(metadata_csv_path)
    
    if limit is not None:
        df_meta = df_meta.iloc[:limit]
        
    filenames = df_meta["filename"].tolist()
    n_mocks = len(filenames)
    n_features = len(k_values) * len(r_bins)
    
    # Pre-allocate feature matrix X to enforce strict index matching
    X = np.zeros((n_mocks, n_features), dtype=np.float32)
    
    print(f"Processing {n_mocks} mocks across {n_workers} workers...")
    print(f"Feature vector length per mock: {n_features}")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_mock,
                mock_path / fname,
                box_size,
                k_values,
                r_bins,
                n_random_points,
                seed=idx
            ): idx
            for idx, fname in enumerate(filenames)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=n_mocks, desc="Extracting k-NN Stats"):
            idx = future_to_idx[future]
            X[idx] = future.result()
            
    np.save(output_feature_path, X)
    print(f"✓ Success! Saved feature matrix X of shape {X.shape} to: {output_feature_path}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    