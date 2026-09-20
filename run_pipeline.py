"""End-to-end command line runner for the InsightForge pipeline.

Examples::

    python run_pipeline.py                       # full run
    python run_pipeline.py --skip-report         # analysis + charts + eval only
    python run_pipeline.py --ask "Which region grew fastest?"
    python run_pipeline.py --provider offline    # force the no-key path

Everything it produces lands in ``outputs/``:

    outputs/figures/*.png              the nine charts
    outputs/reports/analysis.json      every computed metric
    outputs/reports/report.md          insights + recommendations + summary
    outputs/reports/evaluation.csv     QAEvalChain results
    outputs/interactions.jsonl         the monitoring log
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# Windows consoles default to cp1252, which cannot print the characters an
# LLM happily returns. Force UTF-8 (and never crash on an odd glyph).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
        pass

from insightforge.config import settings
from insightforge.data_loader import load_sales_data
from insightforge.evaluation import build_eval_set, evaluate, save_eval_set, save_report
from insightforge.knowledge_base import KnowledgeBase
from insightforge.monitoring import usage_report
from insightforge.rag_chain import InsightForgeAssistant
from insightforge.visualization import save_all_figures


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the InsightForge BI pipeline.")
    parser.add_argument("--data", default=str(settings.data_file), help="path to the sales CSV")
    parser.add_argument("--provider", default=None, help="openrouter | openai | google | offline")
    parser.add_argument("--ask", action="append", default=[], help="question to ask (repeatable)")
    parser.add_argument("--no-pdfs", action="store_true", help="exclude the reference PDFs")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--insights", type=int, default=5)
    args = parser.parse_args(argv)

    started = time.perf_counter()

    # --- 1-2. data preparation + knowledge base ---------------------------
    banner("1-2 | Data preparation and knowledge base")
    df = load_sales_data(args.data)
    kb = KnowledgeBase(df, include_pdfs=not args.no_pdfs)
    kb.build_vector_store()
    print(json.dumps(kb.describe(), indent=2))
    kb.save()
    (settings.reports_dir / "analysis.json").write_text(
        json.dumps(kb.summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"metrics written to {settings.reports_dir / 'analysis.json'}")

    # --- 3-6. assistant ----------------------------------------------------
    banner("3-6 | LLM application, RAG, chained prompts and memory")
    assistant = InsightForgeAssistant(kb, provider=args.provider)
    print(f"provider: {assistant.provider}")

    questions = args.ask or [
        "Give me an overview of business performance.",
        "Which product generates the most revenue and what share of the total is that?",
        "And how does that product do in the West region?",  # tests memory
    ]
    for question in questions:
        result = assistant.ask(question)
        print(f"\nQ: {question}")
        if result["standalone_question"] != question:
            print(f"   (condensed to: {result['standalone_question']})")
        print(f"A: {result['answer']}")
        print(
            f"   [{result['provider']} | {result['latency_s']}s | "
            f"{len(result['sources'])} sources]"
        )

    # --- 7a. visualisations ------------------------------------------------
    if not args.skip_figures:
        banner("7a | Visualisations")
        for path in save_all_figures(df):
            print(f"  wrote {path.relative_to(settings.project_root)}")

    # --- 7b. chained report ------------------------------------------------
    if not args.skip_report:
        banner("7b | Chained report: insights -> recommendations -> summary")
        report = assistant.full_report(args.insights)
        markdown = (
            "# InsightForge report\n\n"
            f"_Generated with {report['provider']} in {report['latency_s']}s_\n\n"
            f"## Executive summary\n\n{report['executive_summary']}\n\n"
            f"## Key insights\n\n{report['insights']}\n\n"
            f"## Recommendations\n\n{report['recommendations']}\n"
        )
        path = settings.reports_dir / "report.md"
        path.write_text(markdown, encoding="utf-8")
        print(report["executive_summary"])
        print(f"\nfull report written to {path}")

    # --- 7c. evaluation ----------------------------------------------------
    if not args.skip_eval:
        banner("7c | Evaluation with QAEvalChain")
        examples = build_eval_set(kb)
        save_eval_set(examples)
        evaluation = evaluate(assistant, examples)
        print(json.dumps(evaluation["metrics"], indent=2))
        print(
            evaluation["table"][
                ["question", "qaeval_grade", "numerically_grounded", "latency_s"]
            ].to_string(index=False)
        )
        save_report(evaluation)
        print(f"\nevaluation written to {settings.reports_dir}")

    # --- monitoring --------------------------------------------------------
    banner("Monitoring summary")
    print(json.dumps(usage_report(), indent=2))
    print(f"\nPipeline finished in {time.perf_counter() - started:.1f}s")
    print("Next step: `streamlit run app.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
