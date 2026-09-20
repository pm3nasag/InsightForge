"""Step 3a - Advanced data summary.

``DataAnalyzer`` turns the enriched dataframe into the four families of
metrics the problem statement asks for:

1. Sales performance by time period (year / quarter / month / weekday)
2. Product and regional analysis (incl. the product x region matrix)
3. Customer segmentation by demographics (age band, gender, satisfaction)
4. Statistical measures (mean, median, std, quartiles, skew, trend, outliers)

Every method returns plain pandas objects; :meth:`DataAnalyzer.summary`
returns a nested, JSON-serialisable dict that both the knowledge base and the
Streamlit UI consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from insightforge.data_loader import dataset_profile


def _round(value: Any, digits: int = 2) -> Any:
    """Numpy-safe rounding that also survives NaN / None."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(np.round(value, digits))
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    """Dataframe -> list of plain-python dicts (safe for ``json.dumps``)."""
    return [
        {str(k): _round(v) for k, v in row.items()}
        for row in df.reset_index().to_dict(orient="records")
    ]


class DataAnalyzer:
    """Computes every metric InsightForge reasons about."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # ------------------------------------------------------------------
    # 1. Sales performance by time period
    # ------------------------------------------------------------------
    def sales_by_period(self, period: str = "Month") -> pd.DataFrame:
        """Aggregate sales for Year / Quarter / Month / Week / Day_Of_Week."""
        if period not in self.df.columns:
            raise ValueError(f"Unknown period column: {period}")
        grouped = (
            self.df.groupby(period, observed=True)
            .agg(
                total_sales=("Sales", "sum"),
                orders=("Sales", "count"),
                avg_sale=("Sales", "mean"),
                median_sale=("Sales", "median"),
                avg_satisfaction=("Customer_Satisfaction", "mean"),
            )
            .round(2)
        )
        if period in {"Year", "Quarter", "Month", "Week"}:
            grouped = grouped.sort_index()
            grouped["growth_vs_prev_pct"] = (
                grouped["total_sales"].pct_change().mul(100).round(2)
            )
        return grouped

    def growth_summary(self, period: str = "Month") -> dict:
        """Best / worst period plus the overall trend direction."""
        totals = self.sales_by_period(period)["total_sales"]
        values = totals.to_numpy(dtype=float)
        slope = (
            float(np.polyfit(np.arange(len(values)), values, 1)[0])
            if len(values) > 1
            else 0.0
        )
        first_half, second_half = np.array_split(values, 2)
        hoh = (
            (second_half.mean() - first_half.mean()) / first_half.mean() * 100
            if len(first_half) and first_half.mean()
            else None
        )
        return {
            "period": period,
            "periods_covered": int(len(totals)),
            "best_period": str(totals.idxmax()),
            "best_period_sales": _round(totals.max()),
            "worst_period": str(totals.idxmin()),
            "worst_period_sales": _round(totals.min()),
            "average_period_sales": _round(totals.mean()),
            "trend_slope_per_period": _round(slope),
            "trend_direction": (
                "upward" if slope > 0 else "downward" if slope < 0 else "flat"
            ),
            "first_half_avg": _round(first_half.mean()) if len(first_half) else None,
            "second_half_avg": _round(second_half.mean()) if len(second_half) else None,
            "half_over_half_change_pct": _round(hoh),
        }

    # ------------------------------------------------------------------
    # 2. Product and regional analysis
    # ------------------------------------------------------------------
    def _by_dimension(self, dimension: str) -> pd.DataFrame:
        grouped = (
            self.df.groupby(dimension, observed=True)
            .agg(
                total_sales=("Sales", "sum"),
                orders=("Sales", "count"),
                avg_sale=("Sales", "mean"),
                median_sale=("Sales", "median"),
                std_sale=("Sales", "std"),
                avg_satisfaction=("Customer_Satisfaction", "mean"),
                avg_customer_age=("Customer_Age", "mean"),
            )
            .round(2)
            .sort_values("total_sales", ascending=False)
        )
        grouped["share_of_sales_pct"] = (
            grouped["total_sales"] / grouped["total_sales"].sum() * 100
        ).round(2)
        return grouped

    def product_performance(self) -> pd.DataFrame:
        return self._by_dimension("Product")

    def regional_performance(self) -> pd.DataFrame:
        return self._by_dimension("Region")

    def product_region_matrix(self) -> pd.DataFrame:
        return (
            self.df.pivot_table(
                index="Product",
                columns="Region",
                values="Sales",
                aggfunc="sum",
                observed=True,
            )
            .fillna(0)
            .round(2)
        )

    # ------------------------------------------------------------------
    # 3. Customer segmentation by demographics
    # ------------------------------------------------------------------
    def customer_segments(self) -> dict[str, pd.DataFrame]:
        return {
            "by_age_group": self._by_dimension("Age_Group"),
            "by_gender": self._by_dimension("Customer_Gender"),
            "by_satisfaction_band": self._by_dimension("Satisfaction_Band"),
        }

    def segment_matrix(self) -> pd.DataFrame:
        """Age band x gender -> average sale value."""
        return (
            self.df.pivot_table(
                index="Age_Group",
                columns="Customer_Gender",
                values="Sales",
                aggfunc="mean",
                observed=True,
            )
            .round(2)
            .fillna(0)
        )

    # ------------------------------------------------------------------
    # 4. Statistical measures
    # ------------------------------------------------------------------
    def statistics(self) -> dict:
        stats: dict[str, Any] = {}
        for col in ("Sales", "Customer_Age", "Customer_Satisfaction"):
            series = self.df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            mean = series.mean()
            stats[col] = {
                "count": int(series.count()),
                "mean": _round(mean),
                "median": _round(series.median()),
                "std": _round(series.std()),
                "variance": _round(series.var()),
                "min": _round(series.min()),
                "max": _round(series.max()),
                "q1": _round(q1),
                "q3": _round(q3),
                "iqr": _round(iqr),
                "skew": _round(series.skew()),
                "kurtosis": _round(series.kurtosis()),
                "coefficient_of_variation_pct": _round(
                    series.std() / mean * 100 if mean else None
                ),
                "outlier_count": int(outliers.count()),
            }

        numeric = self.df[["Sales", "Customer_Age", "Customer_Satisfaction"]]
        stats["correlations"] = {
            f"{a} ~ {b}": _round(numeric[a].corr(numeric[b]))
            for a, b in (
                ("Sales", "Customer_Age"),
                ("Sales", "Customer_Satisfaction"),
                ("Customer_Age", "Customer_Satisfaction"),
            )
        }
        return stats

    # ------------------------------------------------------------------
    # Aggregate views
    # ------------------------------------------------------------------
    def kpis(self) -> dict:
        df = self.df
        products = self.product_performance()
        regions = self.regional_performance()
        monthly = self.sales_by_period("Month")["total_sales"]
        return {
            "total_sales": _round(df["Sales"].sum()),
            "total_orders": int(len(df)),
            "average_order_value": _round(df["Sales"].mean()),
            "median_order_value": _round(df["Sales"].median()),
            "sales_std_dev": _round(df["Sales"].std()),
            "average_satisfaction": _round(df["Customer_Satisfaction"].mean()),
            "average_customer_age": _round(df["Customer_Age"].mean()),
            "top_product": str(products.index[0]),
            "top_product_sales": _round(products["total_sales"].iloc[0]),
            "bottom_product": str(products.index[-1]),
            "bottom_product_sales": _round(products["total_sales"].iloc[-1]),
            "top_region": str(regions.index[0]),
            "top_region_sales": _round(regions["total_sales"].iloc[0]),
            "bottom_region": str(regions.index[-1]),
            "bottom_region_sales": _round(regions["total_sales"].iloc[-1]),
            "best_month": str(monthly.idxmax()),
            "best_month_sales": _round(monthly.max()),
            "worst_month": str(monthly.idxmin()),
            "worst_month_sales": _round(monthly.min()),
            "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        }

    def summary(self) -> dict:
        """One nested dict holding every metric - the analytical backbone."""
        segments = self.customer_segments()
        return {
            "profile": dataset_profile(self.df),
            "kpis": self.kpis(),
            "time": {
                period: _records(self.sales_by_period(period))
                for period in ("Year", "Quarter", "Month", "Day_Of_Week")
            },
            "growth": {
                period: self.growth_summary(period)
                for period in ("Year", "Quarter", "Month")
            },
            "product": _records(self.product_performance()),
            "region": _records(self.regional_performance()),
            "product_region_matrix": self.product_region_matrix().to_dict(
                orient="index"
            ),
            "segments": {name: _records(table) for name, table in segments.items()},
            "age_gender_avg_sale": self.segment_matrix().to_dict(orient="index"),
            "statistics": self.statistics(),
        }
