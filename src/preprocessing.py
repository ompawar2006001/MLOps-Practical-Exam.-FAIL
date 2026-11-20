"""Data ingestion and cleaning for the Breast Cancer pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Paths and constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "breast_cancer_cleaned.csv"

ENTITY_KEY = "patient_id"
TIMESTAMP_KEY = "event_timestamp"


def run_preprocessing() -> Path:
    """Load, clean, and save the Breast Cancer dataset."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cancer = load_breast_cancer(as_frame=True)
    df: pd.DataFrame = cancer.frame.copy()

    logger.info("Loaded raw Breast Cancer dataset with shape %s", df.shape)

    # Sanitize column names
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Remove duplicates and null values
    n_before = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    logger.info(
        "Dropped %d duplicate/null rows (kept %d)",
        n_before - len(df),
        len(df),
    )

    # Add entity key and timestamp
    df.insert(0, ENTITY_KEY, range(1001, 1001 + len(df)))
    df[TIMESTAMP_KEY] = pd.Timestamp.now(tz="UTC")

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    logger.info(
        "Saved cleaned dataset to %s | Shape: %s",
        OUTPUT_PATH,
        df.shape,
    )

    return OUTPUT_PATH


def main() -> None:
    """CLI entry point."""
    run_preprocessing()


if __name__ == "__main__":
    main()