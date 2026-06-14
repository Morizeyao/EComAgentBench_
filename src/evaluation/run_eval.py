"""评测入口：info_source 标注和指标计算。"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.database.product_db import ProductDB
from src.evaluation.judge import RubricJudge
from src.evaluation.metrics import compute_metrics
from src.llm import LLMClient, resolve_llm_config, resolve_prediction_mode_config

logger = logging.getLogger(__name__)


def _attach_rubric_metadata(rubrics: list[dict], results: list[dict]) -> list[dict]:
    rubric_index = {rubric["id"]: rubric for rubric in rubrics}
    enriched: list[dict] = []
    for item in results:
        rubric_id = str(item.get("rubric_id", ""))
        source = rubric_index.get(rubric_id, {})
        enriched.append({
            **item,
            "type": source.get("type", "unknown"),
            "field": source.get("field", ""),
            "description": source.get("description", ""),
            "info_source": source.get("info_source", "query"),
        })
    return enriched


def _summarize_failed_rubrics(flat_rubrics: list[dict]) -> tuple[list[str], list[str], list[str]]:
    failed_ids = []
    failed_types = []
    failed_reasons = []
    for rubric in flat_rubrics:
        if rubric.get("satisfied"):
            continue
        failed_ids.append(str(rubric.get("rubric_id", "")))
        failed_types.append(str(rubric.get("type", "unknown")))
        reason = str(rubric.get("reasoning", "")).strip()
        if reason:
            failed_reasons.append(reason)
    return failed_ids, failed_types, failed_reasons


def _evaluate_one_sample(pred: dict, benchmark: dict,
                         db: ProductDB, judge: RubricJudge) -> dict:
    sid = pred["id"]

    predicted_product_id = pred.get("predicted_product_id")
    recommended_product = None
    if predicted_product_id:
        p = db.get_product(predicted_product_id)
        if p:
            recommended_product = {
                "product_id": predicted_product_id,
                "title": p.get("title", ""),
                "store": p.get("store", ""),
                "average_rating": p.get("average_rating"),
                "rating_number": p.get("rating_number"),
                "price": p.get("price"),
                "details": p.get("details", {}),
                "features": p.get("features", []),
            }

    rubrics = benchmark.get("rubrics", [])

    has_review_rubrics = any(
        rb.get("type") == "review_opinion" for rb in rubrics
    )
    if has_review_rubrics and recommended_product:
        recommended_product["reviews"] = db.get_reviews(recommended_product["product_id"], limit=20)

    judge_output = judge.judge_sample(
        recommended_product, rubrics, benchmark.get("voucher"),
    )

    results = _attach_rubric_metadata(rubrics, judge_output["results"])

    all_satisfied = all(rb.get("satisfied") for rb in results) if results else False
    satisfied_count = sum(1 for rb in results if rb.get("satisfied"))
    total_rubrics = len(results)
    satisfaction_rate = satisfied_count / total_rubrics if total_rubrics else 0
    failed_rubric_ids, failed_rubric_types, failed_reasons = _summarize_failed_rubrics(results)

    target_product = benchmark.get("target_product", {})
    target_product_id = target_product.get("product_id", "")
    exact_product_match = predicted_product_id == target_product_id if predicted_product_id else False

    return {
        "id": sid,
        "intent_category": pred.get("intent_category", ""),
        "rubric_results": results,
        "all_satisfied": all_satisfied,
        "exact_product_match": exact_product_match,
        "satisfaction_rate": round(satisfaction_rate, 4),
        "judge_model": judge.model,
        "predicted_product_id": predicted_product_id,
        "target_product_id": target_product_id,
        "prediction_finished": bool(pred.get("finished", False)),
        "total_tool_calls": pred.get("total_tool_calls", 0),
        "ask_user_calls": pred.get("ask_user_calls", 0),
        "get_profile_calls": pred.get("get_profile_calls", 0),
        "runtime_error": False,
        "error_message": "",
        "failed_rubric_count": len(failed_rubric_ids),
        "failed_rubric_ids": failed_rubric_ids,
        "failed_rubric_types": sorted(set(failed_rubric_types)),
        "failed_reasons": failed_reasons,
    }


def _evaluate_one_sample_safe(pred: dict, benchmark: dict,
                              db: ProductDB, judge: RubricJudge) -> dict:
    try:
        return _evaluate_one_sample(pred, benchmark, db, judge)
    except Exception as exc:
        logger.exception("Failed to evaluate %s", pred.get("id"))
        target_product = benchmark.get("target_product", {})
        target_product_id = target_product.get("product_id", "")
        predicted_product_id = pred.get("predicted_product_id")
        return {
            "id": pred.get("id", ""),
            "intent_category": pred.get("intent_category", ""),
            "rubric_results": [],
            "all_satisfied": False,
            "exact_product_match": predicted_product_id == target_product_id if predicted_product_id else False,
            "satisfaction_rate": 0.0,
            "judge_model": judge.model,
            "predicted_product_id": predicted_product_id,
            "target_product_id": target_product_id,
            "prediction_finished": bool(pred.get("finished", False)),
            "total_tool_calls": pred.get("total_tool_calls", 0),
            "ask_user_calls": pred.get("ask_user_calls", 0),
            "get_profile_calls": pred.get("get_profile_calls", 0),
            "runtime_error": True,
            "error_message": str(exc),
            "failed_rubric_count": 0,
            "failed_rubric_ids": [],
            "failed_rubric_types": [],
            "failed_reasons": [],
        }


def _render_summary_markdown(summary: dict) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        "## Overall",
        "",
        f"- Model: `{summary.get('model', '')}`",
        f"- Judge model: `{summary.get('judge_model', '')}`",
        f"- Total samples: {summary.get('total_samples', 0)}",
        f"- Evaluated samples: {summary.get('evaluated_samples', 0)}",
        f"- Runtime errors: {summary.get('runtime_error_count', 0)} ({summary.get('runtime_error_rate', 0)})",
        f"- **Overall accuracy: {summary.get('overall_accuracy', 0)}** (exact_match OR judge_correct)",
        f"- Exact product match correct: {summary.get('exact_match_correct', 0)} ({summary.get('exact_match_rate', 0)})",
        f"- Non-exact but judge correct: {summary.get('non_exact_but_correct', 0)} ({summary.get('non_exact_but_correct_rate', 0)})",
        f"- Judge-only correct (all_satisfied): {summary.get('judge_only_correct', 0)}",
        f"- Judge misjudged (exact match but judge failed): {summary.get('judge_misjudged_count', 0)}",
        f"- Overall rubric satisfaction: {summary.get('overall_rubric_satisfaction', 0)}",
        f"- Prediction finished rate: {summary.get('prediction_finished_rate', 0)}",
        f"- Average tool calls: {summary.get('avg_tool_calls', 0)}",
        f"- Median tool calls: {summary.get('median_tool_calls', 0)}",
        "",
        "## By Info Source",
        "",
        "| source | count | satisfied | satisfaction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source in ("query", "persona", "clarification"):
        metrics = summary.get("by_source", {}).get(source, {})
        lines.append(
            f"| {source} | {metrics.get('count', 0)} | "
            f"{metrics.get('satisfied', 0)} | {metrics.get('satisfaction', 0)} |"
        )

    persona = summary.get("persona_utilization", {})
    clarification = summary.get("clarification_efficiency", {})
    lines.extend([
        "",
        "## Persona & Clarification",
        "",
        f"- Profile called rate: {persona.get('profile_called_rate', 0)}",
        f"- Avg ask_user calls: {clarification.get('avg_ask_user_calls', 0)}",
        f"- Total clarification slots: {clarification.get('total_clarification_slots', 0)}",
    ])

    lines.extend([
        "",
        "## By Intent",
        "",
        "| intent | count | evaluated | accuracy | exact_match | non_exact_correct | rubric_satisfaction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for intent_id, metrics in sorted(summary.get("by_intent", {}).items()):
        lines.append(
            f"| {intent_id} | {metrics.get('count', 0)} | {metrics.get('evaluated_samples', 0)} | "
            f"{metrics.get('accuracy', 0)} | {metrics.get('exact_match_correct', 0)} | "
            f"{metrics.get('non_exact_but_correct', 0)} | "
            f"{metrics.get('rubric_satisfaction', 0)} |"
        )

    lines.extend([
        "",
        "## By Rubric Type",
        "",
        "| rubric_type | count | satisfied | satisfaction |",
        "| --- | ---: | ---: | ---: |",
    ])
    for rubric_type, metrics in sorted(summary.get("by_rubric_type", {}).items()):
        lines.append(
            f"| {rubric_type} | {metrics.get('count', 0)} | {metrics.get('satisfied', 0)} | {metrics.get('satisfaction', 0)} |"
        )

    failure_reasons = summary.get("top_failure_reasons", {})
    if failure_reasons:
        lines.extend(["", "## Top Failure Reasons", ""])
        for reason, count in failure_reasons.items():
            lines.append(f"- {count}: {reason}")

    runtime_examples = summary.get("runtime_error_examples", [])
    if runtime_examples:
        lines.extend(["", "## Runtime Error Examples", ""])
        for item in runtime_examples:
            lines.append(
                f"- `{item.get('id', '')}` ({item.get('intent_category', '')}): {item.get('error_message', '')}"
            )

    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    config_path: str = "configs/settings.yaml",
    predictions_path: str | None = None,
):
    load_dotenv()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    eval_config = config.get("evaluation", {})
    threads = eval_config.get("num_threads", config.get("num_threads", 16))

    benchmark_dir = data_config["benchmark_dir"]
    benchmark_path = os.path.join(benchmark_dir, "benchmark.jsonl")

    if predictions_path is None:
        _, _, pred_llm_config = resolve_prediction_mode_config(config)
        pred_model = pred_llm_config.get("model", "unknown")
        safe_pred_model = pred_model.replace("/", "_")
        predictions_file = Path(benchmark_dir) / f"predictions_{safe_pred_model}.jsonl"
    else:
        predictions_file = Path(predictions_path)

    if not predictions_file.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    output_dir = predictions_file.parent
    output_stem = f"{predictions_file.stem}_evaluation"
    logger.info("Using predictions file: %s", predictions_file)

    logger.info("Loading product database...")
    db = ProductDB(data_config["db_path"])

    benchmarks = {}
    skipped_invalid = 0
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s.get("sample_status", "ok") != "ok":
                skipped_invalid += 1
                continue
            benchmarks[s["id"]] = s
    logger.info(
        "Loaded %d valid benchmark samples (skipped %d)",
        len(benchmarks), skipped_invalid,
    )

    predictions = []
    with open(predictions_file, encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))
    logger.info("Loaded %d predictions", len(predictions))

    judge_llm_config = resolve_llm_config(config, section="evaluation")
    judge_llm = LLMClient(
        mode=judge_llm_config.get("provider", "openai"),
        config=judge_llm_config,
    )
    judge = RubricJudge(llm_client=judge_llm)

    tasks = []
    model_name = predictions[0].get("model", "unknown") if predictions else "unknown"
    for pred in predictions:
        benchmark = benchmarks.get(pred["id"])
        if not benchmark:
            logger.warning("No benchmark found for %s", pred["id"])
            continue
        tasks.append((pred, benchmark))

    logger.info("Running evaluation with %d threads...", threads)
    eval_results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(_evaluate_one_sample_safe, pred, bm, db, judge): pred["id"]
            for pred, bm in tasks
        }
        for future in as_completed(futures):
            sid = futures[future]
            eval_results.append(future.result())
            logger.info("[%d/%d] Evaluated %s", len(eval_results), len(tasks), sid)

    eval_results.sort(key=lambda r: r["id"])

    pred_index = {p["id"]: p for p in predictions}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{output_stem}_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in eval_results:
            pred = pred_index.get(r["id"], {})
            merged = {**pred, **r}
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")

    summary = compute_metrics(eval_results, benchmarks)
    summary["model"] = model_name
    summary["judge_model"] = judge.model
    summary_path = output_dir / f"{output_stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary_md_path = output_dir / f"{output_stem}_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(_render_summary_markdown(summary))

    logger.info("Evaluation results: %s", results_path)
    logger.info("Summary JSON: %s", summary_path)
    logger.info("Summary Markdown: %s", summary_md_path)
    logger.info("Overall accuracy: %s", summary["overall_accuracy"])
    token_usage = judge_llm.get_token_usage()
    logger.info(
        "Total LLM tokens: input=%d thinking=%d output=%d",
        token_usage["input_tokens"],
        token_usage["thinking_tokens"],
        token_usage["output_tokens"],
    )

    return eval_results, summary


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()
    run_evaluation(args.config, args.predictions)
