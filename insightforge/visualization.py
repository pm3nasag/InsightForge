"""Step 7 - Data visualisation.

The problem statement asks for four families of chart:

1. Sales trends over time
2. Product performance comparisons
3. Regional analysis
4. Customer demographics and segmentation

Each builder returns a matplotlib ``Figure`` so it can be rendered by
Streamlit (``st.pyplot``), shown inline in the notebook, or written to
``outputs/figures`` by :func:`save_all_figures`.  A shared palette and helper
keep the whole deck visually consistent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe in headless / Streamlit contexts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from insightforge.analysis import DataAnalyzer
from insightforge.config import settings

PALETTE = ["#2F6F8F", "#E08D3C", "#5E9C76", "#B5546A", "#7A6BA8", "#8C8C8C"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update(
    {
        "figure.autolayout": True,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#D5D8DC",
        "font.size": 9,
    }
)


def _finish(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Sales trends over time
# ---------------------------------------------------------------------------
def plot_sales_trend(df: pd.DataFrame, period: str = "Month"):
    """Total sales per period with a rolling mean and a linear trend line."""
    series = DataAnalyzer(df).sales_by_period(period)["total_sales"]
    x = np.arange(len(series))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, series.to_numpy(), color=PALETTE[0], linewidth=1.8, label="Total sales")
    if len(series) >= 4:
        window = max(2, len(series) // 12)
        ax.plot(
            x,
            series.rolling(window, min_periods=1).mean().to_numpy(),
            color=PALETTE[1],
            linewidth=2.2,
            linestyle="--",
            label=f"{window}-period rolling mean",
        )
    if len(series) > 1:
        slope, intercept = np.polyfit(x, series.to_numpy(dtype=float), 1)
        ax.plot(x, slope * x + intercept, color=PALETTE[3], linewidth=1.2, label="Linear trend")

    step = max(1, len(series) // 14)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(label) for label in series.index[::step]], rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=8)
    _finish(ax, f"Sales trend by {period.lower()}", period, "Total sales")
    return fig


def plot_seasonality(df: pd.DataFrame):
    """Average sales by calendar month and by weekday - the seasonal view."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    monthly = df.groupby(df["Date"].dt.month)["Sales"].mean()
    axes[0].bar(monthly.index, monthly.to_numpy(), color=PALETTE[0])
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(
        ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    )
    _finish(axes[0], "Average sale by calendar month", "Month", "Average sale")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = df.groupby("Day_Of_Week")["Sales"].mean().reindex(order)
    axes[1].bar(range(len(weekday)), weekday.to_numpy(), color=PALETTE[2])
    axes[1].set_xticks(range(len(weekday)))
    axes[1].set_xticklabels([d[:3] for d in order])
    _finish(axes[1], "Average sale by weekday", "Day of week", "Average sale")
    return fig


# ---------------------------------------------------------------------------
# 2. Product performance comparisons
# ---------------------------------------------------------------------------
def plot_product_performance(df: pd.DataFrame):
    """Revenue by product plus the distribution of order values behind it."""
    analyzer = DataAnalyzer(df)
    products = analyzer.product_performance()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].barh(products.index[::-1], products["total_sales"][::-1], color=PALETTE[0])
    for i, (name, value) in enumerate(zip(products.index[::-1], products["total_sales"][::-1])):
        share = products.loc[name, "share_of_sales_pct"]
        axes[0].text(value, i, f"  {value:,.0f} ({share:.1f}%)", va="center", fontsize=8)
    axes[0].set_xlim(0, products["total_sales"].max() * 1.25)
    _finish(axes[0], "Total revenue by product", "Total sales", "")

    order = list(products.index)
    sns.boxplot(
        data=df,
        x="Product",
        y="Sales",
        hue="Product",
        order=order,
        hue_order=order,
        palette=PALETTE[: len(order)],
        legend=False,
        ax=axes[1],
    )
    _finish(axes[1], "Order value distribution by product", "Product", "Sale value")
    axes[1].tick_params(axis="x", rotation=20)
    return fig


def plot_product_trend(df: pd.DataFrame):
    """Monthly revenue per product - who is gaining and who is fading."""
    pivot = df.pivot_table(
        index="Month", columns="Product", values="Sales", aggfunc="sum", observed=True
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, product in enumerate(pivot.columns):
        ax.plot(
            range(len(pivot)),
            pivot[product].to_numpy(),
            label=str(product),
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.6,
        )
    step = max(1, len(pivot) // 14)
    ax.set_xticks(range(0, len(pivot), step))
    ax.set_xticklabels(list(pivot.index)[::step], rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=8, ncol=len(pivot.columns))
    _finish(ax, "Monthly revenue by product", "Month", "Total sales")
    return fig


# ---------------------------------------------------------------------------
# 3. Regional analysis
# ---------------------------------------------------------------------------
def plot_regional_analysis(df: pd.DataFrame):
    """Regional revenue, regional satisfaction, and the product x region heatmap."""
    analyzer = DataAnalyzer(df)
    regions = analyzer.regional_performance()
    matrix = analyzer.product_region_matrix()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].bar(regions.index, regions["total_sales"], color=PALETTE[0])
    for i, value in enumerate(regions["total_sales"]):
        axes[0].text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, regions["total_sales"].max() * 1.15)
    _finish(axes[0], "Revenue by region", "Region", "Total sales")

    axes[1].bar(regions.index, regions["avg_satisfaction"], color=PALETTE[2])
    axes[1].set_ylim(0, 5)
    for i, value in enumerate(regions["avg_satisfaction"]):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    _finish(axes[1], "Average satisfaction by region", "Region", "Satisfaction (1-5)")

    sns.heatmap(
        matrix, annot=True, fmt=",.0f", cmap="Blues", cbar=False, ax=axes[2],
        annot_kws={"fontsize": 8},
    )
    _finish(axes[2], "Product x region revenue", "Region", "Product")
    return fig


# ---------------------------------------------------------------------------
# 4. Customer demographics and segmentation
# ---------------------------------------------------------------------------
def plot_customer_demographics(df: pd.DataFrame):
    """Age distribution, revenue by age band, gender split, satisfaction mix."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    axes[0][0].hist(df["Customer_Age"].dropna(), bins=20, color=PALETTE[0], edgecolor="white")
    _finish(axes[0][0], "Customer age distribution", "Age", "Customers")

    age_order = [g for g in ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
                 if g in set(df["Age_Group"])]
    by_age = df.groupby("Age_Group", observed=True)["Sales"].sum().reindex(age_order)
    axes[0][1].bar(by_age.index, by_age.to_numpy(), color=PALETTE[1])
    _finish(axes[0][1], "Revenue by age group", "Age group", "Total sales")

    by_gender = df.groupby("Customer_Gender", observed=True)["Sales"].sum()
    axes[1][0].pie(
        by_gender.to_numpy(),
        labels=list(by_gender.index),
        autopct="%1.1f%%",
        colors=PALETTE[: len(by_gender)],
        wedgeprops={"edgecolor": "white"},
    )
    axes[1][0].set_title("Revenue share by gender", fontweight="bold")

    band_order = ["Detractor", "Neutral", "Promoter"]
    bands = (
        df.groupby("Satisfaction_Band", observed=True)["Sales"]
        .agg(["count", "sum"])
        .reindex([b for b in band_order if b in set(df["Satisfaction_Band"])])
    )
    axes[1][1].bar(bands.index, bands["count"], color=PALETTE[4])
    for i, (count, total) in enumerate(zip(bands["count"], bands["sum"])):
        axes[1][1].text(i, count, f"{total:,.0f}", ha="center", va="bottom", fontsize=8)
    _finish(axes[1][1], "Customers by satisfaction band\n(labels = revenue)", "Band", "Customers")
    return fig


def plot_segment_heatmap(df: pd.DataFrame):
    """Average order value for every age band x gender cell."""
    matrix = DataAnalyzer(df).segment_matrix()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt=",.0f", cmap="YlGnBu", cbar_kws={"label": "Avg sale"}, ax=ax)
    _finish(ax, "Average order value by age group and gender", "Gender", "Age group")
    return fig


def plot_correlation_matrix(df: pd.DataFrame):
    """Correlations between the numeric fields."""
    numeric = df[["Sales", "Customer_Age", "Customer_Satisfaction"]]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    sns.heatmap(
        numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax
    )
    _finish(ax, "Correlation matrix", "", "")
    return fig


# ---------------------------------------------------------------------------
# Registry / export
# ---------------------------------------------------------------------------
FIGURE_BUILDERS = {
    "sales_trend_month": lambda df: plot_sales_trend(df, "Month"),
    "sales_trend_quarter": lambda df: plot_sales_trend(df, "Quarter"),
    "seasonality": plot_seasonality,
    "product_performance": plot_product_performance,
    "product_trend": plot_product_trend,
    "regional_analysis": plot_regional_analysis,
    "customer_demographics": plot_customer_demographics,
    "segment_heatmap": plot_segment_heatmap,
    "correlation_matrix": plot_correlation_matrix,
}


def save_all_figures(df: pd.DataFrame, directory: Path | None = None) -> list[Path]:
    """Render every chart to PNG - used by the CLI pipeline and the report."""
    directory = Path(directory) if directory is not None else settings.figures_dir
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in FIGURE_BUILDERS.items():
        fig = builder(df)
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
