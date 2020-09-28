import datetime
import numpy as np
import os
import pandas as pd
import shutil
import urllib.request

__all__ = [
    'fetch_ml_ratings',
]

VARIANTS = {
    'ml100k': {'filename': '../datasets/ml/ml-100k.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5},
    'ml1m': {'filename': '../datasets/ml/ml-1m.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5},
    'ml10m': {'filename': '../datasets/ml/ml-10m.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5},
    'ml20m': {'filename': '../datasets/ml/ml-20m.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5},
    'ml25m': {'filename': '../datasets/ml/ml-25m.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5},
    'bx': {'filename': '../datasets/bx/book_crossing.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': "string", 'min_rating': 1, 'max_rating': 10},
    'netflix': {'filename': '../datasets/netflix/netflix.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5},
    'netflix_probe': {'filename': '../datasets/netflix_probe/netflix.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5},
    'netflix_training': {'filename': '../datasets/netflix_training/netflix.csv', 'sep': ';', 'u_type': np.uint32, 'i_type': np.uint32, 'min_rating': 1, 'max_rating': 5}
}

def ml_ratings_csv_to_df(csv_path, variant):
    names = ['u_id', 'i_id', 'rating']
    dtype = {'u_id': VARIANTS[variant]['u_type'], 'i_id': VARIANTS[variant]['i_type'], 'rating': np.float64}

    df = pd.read_csv(csv_path, names=names, dtype=dtype, header=0,
                     sep=VARIANTS[variant]['sep'], engine='python')

    df.reset_index(drop=True, inplace=True)

    return df
pass


def fetch_ml_ratings(variant):
    """Fetches ratings dataset.

    Args:
        data_dir_path (string): explicit data directory path to ratings file.
        variant (string): movie lens dataset variant, could be any of
            ['100k', '1m', '10m', '20m', '25m', 'bx', 'netflix', 'netflix_probe', 'netflix_training'].

    Returns:
        df (pandas.DataFrame): containing the dataset.
    """
    csv_path = VARIANTS[variant]['filename']

    if os.path.exists(csv_path):
        # Return data loaded into a DataFrame
        df = ml_ratings_csv_to_df(csv_path, variant)
        return df
    else:
        print('unexpected')
    pass
pass