import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
    Faithfulness,
)
def build_parser():
    script_path = Path(__file__).resolve()
    results_dir = script_path.parent / "results"
    parser = argparse.ArgumentParser(
        description="Run Ragas evaluation against the latest Node evaluation payload."
    )
    parser.add_argument(
        "--input",
        default=str(results_dir / "latest.json"),
        help="Path to a Node evaluation JSON file.",
    )
    parser.add_argument(
        "--output-json",
        default=str(results_dir / "latest-ragas.json"),
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default=str(results_dir / "latest-ragas.md"),
        help="Path to write the Markdown report.",
    )
    parser.add_argument(
        "--include-abstain",
        action="store_true",
        help="Include abstain cases in answer-only metrics. By default they are skipped.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Override the LLM used by Ragas metrics. Defaults to RAGAS_EVAL_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Override the embedding model used by AnswerRelevancy.",
    )
    return parser


def load_env_file(env_path: Path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def unwrap_score(result):
    value = getattr(result, "value", result)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return round(value, 4)
    raise TypeError(f"Unsupported score result type: {type(value)!r}")


def average(values):
    valid_values = [value for value in values if isinstance(value, (int, float))]
    if not valid_values:
        return None
    return round(sum(valid_values) / len(valid_values), 4)


def build_markdown(report):
    lines = [
        "# Ragas Evaluation",
        "",
        f"- Created: `{report['createdAt']}`",
        f"- Input file: `{report['inputFile']}`",
        f"- Source run ID: `{report['sourceRunId']}`",
        f"- Judge model: `{report['models']['judge']}`",
        f"- Embedding model: `{report['models']['embedding']}`",
        f"- Eligible cases: `{report['summary']['eligibleCases']}` / `{report['summary']['totalCases']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Average |",
        "| --- | ---: |",
    ]

    metric_labels = {
        "answer_relevancy": "Answer relevancy",
        "faithfulness": "Faithfulness",
        "context_utilization": "Context utilization",
        "context_precision": "Context precision",
        "context_recall": "Context recall",
    }

    for metric_key, label in metric_labels.items():
        lines.append(f"| {label} | {report['summary']['metrics'].get(metric_key)} |")

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case | Type | Answer Relevancy | Faithfulness | Context Utilization | Context Precision | Context Recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for case in report["cases"]:
        scores = case["scores"]
        lines.append(
            f"| {case['caseId']} | {case['type']} | {scores.get('answer_relevancy')} | {scores.get('faithfulness')} | {scores.get('context_utilization')} | {scores.get('context_precision')} | {scores.get('context_recall')} |"
        )

    if report["skippedCases"]:
        lines.extend(["", "## Skipped Cases", ""])
        for skipped in report["skippedCases"]:
            lines.append(f"- `{skipped['caseId']}`: {skipped['reason']}")

    return "\n".join(lines) + "\n"


async def score_case(case, scorers):
    sample = case["ragasSample"]
    user_input = sample["user_input"]
    response = sample["response"]
    retrieved_contexts = sample["retrieved_contexts"]
    reference = sample.get("reference")
    scores = {}
    errors = {}

    metric_specs = [
        (
            "answer_relevancy",
            scorers["answer_relevancy"],
            bool(response),
            lambda: {
                "user_input": user_input,
                "response": response,
            },
        ),
        (
            "faithfulness",
            scorers["faithfulness"],
            bool(response and retrieved_contexts),
            lambda: {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            },
        ),
        (
            "context_utilization",
            scorers["context_utilization"],
            bool(response and retrieved_contexts),
            lambda: {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            },
        ),
        (
            "context_precision",
            scorers["context_precision"],
            bool(reference and retrieved_contexts),
            lambda: {
                "user_input": user_input,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            },
        ),
        (
            "context_recall",
            scorers["context_recall"],
            bool(reference and retrieved_contexts),
            lambda: {
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
            },
        ),
    ]

    for metric_name, scorer, enabled, kwargs_factory in metric_specs:
        if not enabled:
            scores[metric_name] = None
            continue

        try:
            scores[metric_name] = unwrap_score(await scorer.ascore(**kwargs_factory()))
        except Exception as exc:  # pragma: no cover - external library/runtime behavior
            scores[metric_name] = None
            errors[metric_name] = str(exc)

    return {
        "caseId": case["id"],
        "type": case["type"],
        "question": case["question"],
        "reference": reference,
        "retrievedContextCount": len(sample["retrieved_contexts"]),
        "scores": scores,
        "errors": errors,
    }


async def main():
    args = build_parser().parse_args()
    script_path = Path(__file__).resolve()
    server_dir = script_path.parent.parent
    load_env_file(server_dir / ".env")

    input_path = Path(args.input).resolve()
    output_json_path = Path(args.output_json).resolve()
    output_md_path = Path(args.output_md).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input evaluation file not found: {input_path}")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Ragas evaluation.")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    source_summary = payload.get("summary", {})
    judge_model = (
        args.judge_model.strip()
        or os.environ.get("RAGAS_EVAL_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    embedding_model = os.environ.get(
        "RAGAS_EMBEDDING_MODEL", ""
    ).strip() or args.embedding_model.strip() or source_summary.get("models", {}).get(
        "embedding", "text-embedding-3-small"
    )

    client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=5)
    llm = llm_factory(judge_model, client=client)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)
    scorers = {
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "faithfulness": Faithfulness(llm=llm),
        "context_utilization": ContextUtilization(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    cases = payload.get("cases", [])
    eligible_cases = []
    skipped_cases = []

    for case in cases:
        sample = case.get("ragasSample") or {}
        retrieved_contexts = sample.get("retrieved_contexts") or []
        should_abstain = bool(case.get("shouldAbstain"))

        if not retrieved_contexts:
            skipped_cases.append(
                {"caseId": case.get("id"), "reason": "No retrieved_contexts were captured."}
            )
            continue

        if should_abstain and not args.include_abstain:
            skipped_cases.append(
                {
                    "caseId": case.get("id"),
                    "reason": "Abstain case skipped by default.",
                }
            )
            continue

        eligible_cases.append(case)

    scored_cases = []
    for case in eligible_cases:
        scored_cases.append(await score_case(case, scorers))

    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "inputFile": str(input_path),
        "sourceRunId": source_summary.get("runId"),
        "models": {
            "judge": judge_model,
            "embedding": embedding_model,
        },
        "summary": {
            "totalCases": len(cases),
            "eligibleCases": len(eligible_cases),
            "metrics": {
                "answer_relevancy": average(
                    case["scores"]["answer_relevancy"] for case in scored_cases
                ),
                "faithfulness": average(
                    case["scores"]["faithfulness"] for case in scored_cases
                ),
                "context_utilization": average(
                    case["scores"]["context_utilization"] for case in scored_cases
                ),
                "context_precision": average(
                    case["scores"]["context_precision"] for case in scored_cases
                ),
                "context_recall": average(
                    case["scores"]["context_recall"] for case in scored_cases
                ),
            },
        },
        "cases": scored_cases,
        "skippedCases": skipped_cases,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_md_path.write_text(build_markdown(report), encoding="utf-8")

    print(f"Ragas evaluation written to {output_json_path}")
    print(f"Markdown report written to {output_md_path}")


if __name__ == "__main__":
    asyncio.run(main())
