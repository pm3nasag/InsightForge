"""Step 7 - Model evaluation with ``QAEvalChain``.

The evaluation set is *derived from the data itself* (:func:`build_eval_set`),
so the ground truth can never drift away from the CSV: every reference answer
is computed with pandas at build time.

Grading runs two complementary checks:

* **QAEvalChain** - LangChain's LLM-as-judge, exactly as the problem statement
  asks, using the custom :data:`~insightforge.prompts.EVAL_PROMPT`.
* **Numeric grounding** - a deterministic check that the key figure from the
  reference answer actually appears in the prediction.  This keeps the score
  meaningful even when the judge is unavailable (offline mode).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from insightforge.config import settings
from insightforge.llm import get_llm
from insightforge.prompts import EVAL_PROMPT

# QAEvalChain lives in `langchain` up to 0.3.x and in `langchain_classic`
# from LangChain 1.0 onwards.
try:  # pragma: no cover - import shim
    from langchain.evaluation.qa import QAEvalChain
except Exception:  # pragma: no cover
    try:
        from langchain_classic.evaluation.qa import QAEvalChain
    except Exception:
        QAEvalChain = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Evaluation set
# ---------------------------------------------------------------------------
def build_eval_set(knowledge_base: Any) -> list[dict]:
    """Build question / ground-truth pairs straight from the computed metrics."""
    summary = knowledge_base.summary
    kpis = summary["kpis"]
    stats = summary["statistics"]
    products = {row["Product"]: row for row in summary["product"]}
    regions = {row["Region"]: row for row in summary["region"]}
    growth_month = summary["growth"]["Month"]
    genders = {row["Customer_Gender"]: row for row in summary["segments"]["by_gender"]}
    ages = summary["segments"]["by_age_group"]
    top_age = max(ages, key=lambda row: row["total_sales"])

    examples = [
        {
            "query": "What is the total revenue in the dataset and how many orders does it cover?",
            "answer": (
                f"Total revenue is {kpis['total_sales']:,.2f} across "
                f"{kpis['total_orders']:,} orders."
            ),
        },
        {
            "query": "Which product generates the most revenue, and how much?",
            "answer": (
                f"{kpis['top_product']} is the top product with "
                f"{kpis['top_product_sales']:,.2f} in revenue "
                f"({products[kpis['top_product']]['share_of_sales_pct']:.2f}% of the total)."
            ),
        },
        {
            "query": "Which product performs worst by revenue?",
            "answer": (
                f"{kpis['bottom_product']} is the weakest product with "
                f"{kpis['bottom_product_sales']:,.2f} in revenue."
            ),
        },
        {
            "query": "Which region has the highest sales?",
            "answer": (
                f"The {kpis['top_region']} region leads with "
                f"{kpis['top_region_sales']:,.2f} in revenue "
                f"({regions[kpis['top_region']]['share_of_sales_pct']:.2f}% of the total)."
            ),
        },
        {
            "query": "What is the average and median order value?",
            "answer": (
                f"The average order value is {kpis['average_order_value']:,.2f} and the "
                f"median order value is {kpis['median_order_value']:,.2f}."
            ),
        },
        {
            "query": "What is the standard deviation of sales?",
            "answer": f"The standard deviation of sales is {stats['Sales']['std']:,.2f}.",
        },
        {
            "query": "Which month had the highest total sales?",
            "answer": (
                f"{kpis['best_month']} was the strongest month with "
                f"{kpis['best_month_sales']:,.2f} in sales."
            ),
        },
        {
            "query": "Is the monthly sales trend going up or down?",
            "answer": (
                f"The monthly trend is {growth_month['trend_direction']}, with a slope of "
                f"{growth_month['trend_slope_per_period']:,.2f} sales per month across "
                f"{growth_month['periods_covered']} months."
            ),
        },
        {
            "query": "How do male and female customers compare in total revenue?",
            "answer": " ".join(
                f"{gender}: {row['total_sales']:,.2f} in revenue from {row['orders']:,} orders."
                for gender, row in genders.items()
            ),
        },
        {
            "query": "Which customer age group spends the most in total?",
            "answer": (
                f"The {top_age['Age_Group']} age group spends the most, with "
                f"{top_age['total_sales']:,.2f} in total revenue across "
                f"{top_age['orders']:,} orders."
            ),
        },
        {
            "query": "What is the average customer satisfaction score?",
            "answer": (
                f"Average customer satisfaction is {kpis['average_satisfaction']:.2f} out of 5."
            ),
        },
        {
            "query": "Is customer satisfaction correlated with sales?",
            "answer": (
                "The correlation between sales and customer satisfaction is "
                f"{stats['correlations']['Sales ~ Customer_Satisfaction']:.2f}, which is "
                "effectively no linear relationship."
            ),
        },
    ]
    return examples


def save_eval_set(examples: list[dict], path: Path | None = None) -> Path:
    path = Path(path) if path is not None else settings.eval_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return path


def load_eval_set(path: Path | None = None) -> list[dict]:
    path = Path(path) if path is not None else settings.eval_file
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    values = []
    for match in NUMBER_RE.findall(text or ""):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


GRADE_RE = re.compile(r"GRADE\s*:\s*(CORRECT|INCORRECT)")


def parse_grade(raw: Any) -> str:
    """Reduce a judge response to CORRECT / INCORRECT / UNKNOWN.

    ``QAEvalChain`` returns the grader's reasoning together with its verdict,
    so the verdict has to be extracted rather than string-matched: note that
    "INCORRECT" contains "CORRECT", which is why it is tested first.
    """
    text = str(raw or "").upper()
    match = GRADE_RE.search(text)
    if match:
        return match.group(1)
    if "INCORRECT" in text:
        return "INCORRECT"
    if "CORRECT" in text:
        return "CORRECT"
    return "UNKNOWN"


def numeric_grounding(reference: str, prediction: str, tolerance: float = 0.02) -> bool:
    """True when every key figure in the reference appears in the prediction."""
    expected = [v for v in _numbers(reference) if abs(v) >= 1]
    if not expected:
        return True
    found = _numbers(prediction)
    if not found:
        return False
    hits = 0
    for value in expected:
        window = max(abs(value) * tolerance, 0.5)
        if any(abs(value - candidate) <= window for candidate in found):
            hits += 1
    return hits >= max(1, len(expected) - 1)


def run_predictions(assistant: Any, examples: list[dict]) -> list[dict]:
    """Answer every evaluation question with memory switched off."""
    predictions = []
    for example in examples:
        result = assistant.ask(example["query"], use_memory=False)
        predictions.append(
            {
                "result": result["answer"],
                "latency_s": result["latency_s"],
                "sources": [s["topic"] for s in result["sources"]],
                "error": result["error"],
            }
        )
    return predictions


def grade_with_qaevalchain(
    examples: list[dict], predictions: list[dict], llm: Any | None = None
) -> list[dict]:
    """Grade predictions with LangChain's ``QAEvalChain`` (LLM-as-judge)."""
    if QAEvalChain is None:
        return [{"results": "SKIPPED - QAEvalChain unavailable"} for _ in examples]

    if llm is None:
        llm, provider = get_llm()
        if provider.startswith("offline"):
            # An extractive model cannot act as a judge; fall back to the
            # deterministic numeric check instead of producing noise.
            return [
                {
                    "results": "CORRECT"
                    if numeric_grounding(ex["answer"], pred["result"])
                    else "INCORRECT",
                    "reasoning": "Graded by numeric grounding (offline mode).",
                }
                for ex, pred in zip(examples, predictions)
            ]

    chain = QAEvalChain.from_llm(llm=llm, prompt=EVAL_PROMPT)
    try:
        return chain.evaluate(
            examples,
            predictions,
            question_key="query",
            answer_key="answer",
            prediction_key="result",
        )
    except Exception as exc:  # pragma: no cover - network failures
        return [{"results": f"ERROR: {exc}"} for _ in examples]


def evaluate(
    assistant: Any, examples: list[dict] | None = None, llm: Any | None = None
) -> dict:
    """Full evaluation run -> per-question table plus headline metrics."""
    examples = examples or build_eval_set(assistant.kb)
    predictions = run_predictions(assistant, examples)
    graded = grade_with_qaevalchain(examples, predictions, llm=llm)

    rows = []
    for example, prediction, grade in zip(examples, predictions, graded):
        raw_grade = grade.get("results", "")
        rows.append(
            {
                "question": example["query"],
                "reference": example["answer"],
                "prediction": prediction["result"],
                "qaeval_grade": parse_grade(raw_grade),
                "qaeval_reasoning": str(raw_grade).strip(),
                "numerically_grounded": numeric_grounding(
                    example["answer"], prediction["result"]
                ),
                "latency_s": prediction["latency_s"],
                "retrieved_topics": ", ".join(prediction["sources"][:4]),
                "error": prediction["error"],
            }
        )

    table = pd.DataFrame(rows)
    correct = int((table["qaeval_grade"] == "CORRECT").sum())
    grounded = int(table["numerically_grounded"].sum())
    total = len(table)

    return {
        "table": table,
        "metrics": {
            "questions": total,
            "qaeval_correct": correct,
            "qaeval_accuracy": round(correct / total, 3) if total else None,
            "numeric_grounding_rate": round(grounded / total, 3) if total else None,
            "avg_latency_s": round(float(table["latency_s"].mean()), 3) if total else None,
            "errors": int(table["error"].notna().sum()),
            "provider": assistant.provider,
        },
    }


def save_report(evaluation: dict, directory: Path | None = None) -> Path:
    """Write the evaluation table and metrics to ``outputs/reports``."""
    directory = Path(directory) if directory is not None else settings.reports_dir
    directory.mkdir(parents=True, exist_ok=True)
    evaluation["table"].to_csv(directory / "evaluation.csv", index=False)
    (directory / "evaluation_metrics.json").write_text(
        json.dumps(evaluation["metrics"], indent=2), encoding="utf-8"
    )
    return directory
