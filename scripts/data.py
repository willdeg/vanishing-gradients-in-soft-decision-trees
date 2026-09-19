from dataclasses import dataclass

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


@dataclass
class DatasetSplit:

    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray


def load_bike_sharing_binary(target_quantile=0.75):
    """
    Load the bike-sharing dataset and convert it into a binary classification task.

    We label an hour as positive when demand is at or above the chosen quantile.
    """

    bike_sharing = fetch_ucirepo(id=275)

    # Work on copies so we do not accidentally modify the raw dataset object
    features = bike_sharing.data.features.copy()
    targets = bike_sharing.data.targets.copy()


    features = features.drop(columns=["dteday"])

    # These columns represent categories rather than continuous
    categorical_cols = ["season", "mnth", "hr", "weekday", "weathersit"]
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True, dtype=float)


    x = features.to_numpy(dtype=np.float64)
    demand = targets["cnt"].to_numpy(dtype=np.float64)
    cutoff = np.quantile(demand, target_quantile)
    y = (demand >= cutoff).astype(np.int64)
    return x, y


def train_val_test_split(
    x,
    y,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
):
    """
    Randomly split the dataset into train, validation, and test sets
    """

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1")

    rng = np.random.default_rng(random_state)
    n_samples = x.shape[0]
    indices = rng.permutation(n_samples)

    train_end = int(train_size * n_samples)
    val_end = train_end + int(val_size * n_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return DatasetSplit(
        train_x=x[train_idx],
        val_x=x[val_idx],
        test_x=x[test_idx],
        train_y=y[train_idx],
        val_y=y[val_idx],
        test_y=y[test_idx],
    )


def standardize_from_train(
    train_x,
    other_arrays,
):
    """
    Standardize arrays using statistics computed on the training split only

    """

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8
    train_x_std = (train_x - mean) / std
    transformed = [(array - mean) / std for array in other_arrays]
    return train_x_std, transformed, mean, std
