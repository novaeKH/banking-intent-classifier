from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_splits(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Keep the original training split; separate model and policy selection."""
    train = pd.read_csv(raw_dir / "train.csv").drop_duplicates()
    test = pd.read_csv(raw_dir / "test.csv")

    def normalize(values):
        return values.str.lower().str.split().str.join(" ")

    overlap = normalize(train["text"]).isin(set(normalize(test["text"])))
    train = train.loc[~overlap].reset_index(drop=True)
    train, validation = train_test_split(
        train, test_size=0.2, stratify=train["category"], random_state=42
    )
    tune, policy = train_test_split(
        validation, test_size=0.5, stratify=validation["category"], random_state=42
    )
    return {
        name: frame.reset_index(drop=True)
        for name, frame in {
            "train": train,
            "tune": tune,
            "policy": policy,
            "test": test,
        }.items()
    }
