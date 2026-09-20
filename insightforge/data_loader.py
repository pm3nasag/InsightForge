"""Step 1 - Data preparation.

The dataset supplied with the capstone is already clean, so this module does
not do any cleaning beyond type coercion.  Its job is to *enrich* the raw
records with the derived columns the analysis layer needs (calendar parts,
customer age bands, satisfaction bands).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insightforge.config import settings

AGE_BINS = [17, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

SATISFACTION_BINS = [0, 2.0, 3.5, 5.0]
SATISFACTION_LABELS = ["Detractor", "Neutral", "Promoter"]

REQUIRED_COLUMNS = [
    "Date",
    "Product",
    "Region",
    "Sales",
    "Customer_Age",
    "Customer_Gender",
    "Customer_Satisfaction",
]


def load_sales_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the sales CSV and add the derived analysis columns."""
    path = Path(path) if path is not None else settings.data_file
    if not path.exists():
        raise FileNotFoundError(
            f"Sales dataset not found at {path}. "
            "Set settings.data_file or pass an explicit path."
        )

    df = pd.read_csv(path, parse_dates=["Date"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    return enrich(df)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and segmentation columns used across the project."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Customer_Age"] = pd.to_numeric(df["Customer_Age"], errors="coerce")
    df["Customer_Satisfaction"] = pd.to_numeric(
        df["Customer_Satisfaction"], errors="coerce"
    ).round(2)

    df = df.dropna(subset=["Date", "Sales"]).sort_values("Date").reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Week"] = df["Date"].dt.to_period("W").astype(str)
    df["Day_Of_Week"] = df["Date"].dt.day_name()

    df["Age_Group"] = pd.cut(
        df["Customer_Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
    ).astype(str)
    df["Satisfaction_Band"] = pd.cut(
        df["Customer_Satisfaction"],
        bins=SATISFACTION_BINS,
        labels=SATISFACTION_LABELS,
        include_lowest=True,
    ).astype(str)

    return df


def dataset_profile(df: pd.DataFrame) -> dict:
    """A compact, JSON-serialisable profile used by the knowledge base."""
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "date_min": df["Date"].min().date().isoformat(),
        "date_max": df["Date"].max().date().isoformat(),
        "products": sorted(df["Product"].dropna().unique().tolist()),
        "regions": sorted(df["Region"].dropna().unique().tolist()),
        "genders": sorted(df["Customer_Gender"].dropna().unique().tolist()),
        "age_groups": [g for g in AGE_LABELS if g in set(df["Age_Group"])],
        "total_sales": float(df["Sales"].sum()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": {
            col: {
                "mean": float(np.round(df[col].mean(), 2)),
                "median": float(np.round(df[col].median(), 2)),
                "std": float(np.round(df[col].std(), 2)),
                "min": float(np.round(df[col].min(), 2)),
                "max": float(np.round(df[col].max(), 2)),
            }
            for col in ("Sales", "Customer_Age", "Customer_Satisfaction")
        },
    }
