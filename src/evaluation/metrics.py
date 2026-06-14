"""评测指标：by_source、persona_utilization、clarification_efficiency。"""

from collections import Counter, defaultdict
from statistics import median


def _flatten_rubric_results(rubric_results: list) -> list[dict]:
    flat = []
    for item in rubric_results:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, dict):
            flat.append(item)
    return flat


def _collect_failed_reason_counts(flat_rubrics: list[dict], max_unique: int = 20) -> dict[str, int]:
    counter = Counter()
    for rubric in flat_rubrics:
        if rubric.get("satisfied"):
            continue
        reason = str(rubric.get("reasoning", "")).strip()
        if reason:
            counter[reason] += 1
    return dict(counter.most_common(max_unique))


def _compute_base_metrics(eval_results: list[dict]) -> dict:
    """计算整体和分类别的评测指标。

    Args:
        eval_results: [{
            "id": "sample_001",
            "intent_category": "...",
            "rubric_results": [[{"rubric_id": "r1", "satisfied": true, ...}, ...], ...],
            "all_satisfied": bool,
            "satisfaction_rate": float,
        }]
    """
    total = len(eval_results)
    if total == 0:
        return {
            "overall_accuracy": 0,
            "overall_rubric_satisfaction": 0,
            "total_samples": 0,
            "evaluated_samples": 0,
            "runtime_error_count": 0,
        }

    runtime_errors = [r for r in eval_results if r.get("runtime_error")]
    valid_results = [r for r in eval_results if not r.get("runtime_error")]

    # 精确匹配 product_id 的样本直接视为完全正确
    exact_match_count = sum(1 for r in valid_results if r.get("exact_product_match"))
    judge_correct_count = sum(1 for r in valid_results if r.get("all_satisfied"))
    # 精确匹配但 judge 认为不满足 rubric 的样本（judge 误判）
    judge_misjudged_count = sum(
        1 for r in valid_results
        if r.get("exact_product_match") and not r.get("all_satisfied")
    )
    # 没有精确匹配但 judge 也认为满足所有 rubric 的样本
    non_exact_but_judge_correct = sum(
        1 for r in valid_results
        if not r.get("exact_product_match") and r.get("all_satisfied")
    )
    # 最终正确 = 精确匹配 ∪ judge认为正确
    all_correct = sum(
        1 for r in valid_results
        if r.get("exact_product_match") or r.get("all_satisfied")
    )

    finished_prediction_count = sum(1 for r in eval_results if r.get("prediction_finished"))
    tool_call_counts = [int(r.get("total_tool_calls", 0) or 0) for r in eval_results]
    total_rubrics = 0
    satisfied_rubrics = 0
    rubric_type_counter = Counter()
    rubric_type_satisfied = Counter()
    all_flat_rubrics: list[dict] = []
    for r in valid_results:
        flat = _flatten_rubric_results(r.get("rubric_results", []))
        all_flat_rubrics.extend(flat)
        for rb in flat:
            total_rubrics += 1
            rubric_type = str(rb.get("type", "unknown"))
            rubric_type_counter[rubric_type] += 1
            if rb.get("satisfied"):
                satisfied_rubrics += 1
                rubric_type_satisfied[rubric_type] += 1

    by_intent = defaultdict(list)
    for r in eval_results:
        by_intent[r["intent_category"]].append(r)

    intent_metrics = {}
    for intent_id, results in by_intent.items():
        intent_runtime_errors = [r for r in results if r.get("runtime_error")]
        valid_intent_results = [r for r in results if not r.get("runtime_error")]
        i_exact = sum(1 for r in valid_intent_results if r.get("exact_product_match"))
        i_judge_correct = sum(1 for r in valid_intent_results if r.get("all_satisfied"))
        correct = sum(
            1 for r in valid_intent_results
            if r.get("exact_product_match") or r.get("all_satisfied")
        )
        i_non_exact_but_correct = sum(
            1 for r in valid_intent_results
            if not r.get("exact_product_match") and r.get("all_satisfied")
        )
        intent_total = 0
        intent_satisfied = 0
        for r in valid_intent_results:
            flat = _flatten_rubric_results(r.get("rubric_results", []))
            for rb in flat:
                intent_total += 1
                if rb.get("satisfied"):
                    intent_satisfied += 1
        n_valid = len(valid_intent_results)
        intent_metrics[intent_id] = {
            "accuracy": round(correct / n_valid, 4) if n_valid else 0,
            "rubric_satisfaction": round(intent_satisfied / intent_total, 4) if intent_total else 0,
            "count": len(results),
            "correct": correct,
            "exact_match_correct": i_exact,
            "non_exact_but_correct": i_non_exact_but_correct,
            "judge_only_correct": i_judge_correct,
            "evaluated_samples": n_valid,
            "runtime_error_count": len(intent_runtime_errors),
        }

    rubric_type_metrics = {}
    for rubric_type, count in rubric_type_counter.items():
        rubric_type_metrics[rubric_type] = {
            "count": count,
            "satisfied": rubric_type_satisfied[rubric_type],
            "satisfaction": round(rubric_type_satisfied[rubric_type] / count, 4) if count else 0,
        }

    runtime_error_examples = [
        {
            "id": r.get("id"),
            "intent_category": r.get("intent_category"),
            "error_message": r.get("error_message", ""),
        }
        for r in runtime_errors[:20]
    ]
    top_failure_reasons = _collect_failed_reason_counts(all_flat_rubrics)

    n_valid = len(valid_results)
    return {
        "overall_accuracy": round(all_correct / n_valid, 4) if n_valid else 0,
        "overall_rubric_satisfaction": round(satisfied_rubrics / total_rubrics, 4) if total_rubrics else 0,
        "total_samples": total,
        "evaluated_samples": n_valid,
        "runtime_error_count": len(runtime_errors),
        "runtime_error_rate": round(len(runtime_errors) / total, 4) if total else 0,
        "total_correct": all_correct,
        "exact_match_correct": exact_match_count,
        "exact_match_rate": round(exact_match_count / n_valid, 4) if n_valid else 0,
        "non_exact_but_correct": non_exact_but_judge_correct,
        "non_exact_but_correct_rate": round(non_exact_but_judge_correct / n_valid, 4) if n_valid else 0,
        "judge_only_correct": judge_correct_count,
        "judge_misjudged_count": judge_misjudged_count,
        "total_rubrics": total_rubrics,
        "satisfied_rubrics": satisfied_rubrics,
        "prediction_finished_count": finished_prediction_count,
        "prediction_unfinished_count": total - finished_prediction_count,
        "prediction_finished_rate": round(finished_prediction_count / total, 4) if total else 0,
        "avg_tool_calls": round(sum(tool_call_counts) / total, 4) if total else 0,
        "median_tool_calls": median(tool_call_counts) if tool_call_counts else 0,
        "by_rubric_type": rubric_type_metrics,
        "top_failure_reasons": top_failure_reasons,
        "runtime_error_examples": runtime_error_examples,
        "by_intent": intent_metrics,
    }


def compute_metrics(eval_results: list[dict], benchmarks: dict | None = None) -> dict:
    """指标计算：含 by_source 和交互指标。"""
    base_metrics = _compute_base_metrics(eval_results)

    # by_source rubric satisfaction
    source_counter: Counter = Counter()
    source_satisfied: Counter = Counter()
    for r in eval_results:
        if r.get("runtime_error"):
            continue
        flat = _flatten_rubric_results(r.get("rubric_results", []))
        for rb in flat:
            source = rb.get("info_source", "query")
            source_counter[source] += 1
            if rb.get("satisfied"):
                source_satisfied[source] += 1

    by_source = {}
    for source in ("query", "persona", "clarification"):
        total = source_counter[source]
        satisfied = source_satisfied[source]
        by_source[source] = {
            "count": total,
            "satisfied": satisfied,
            "satisfaction": round(satisfied / total, 4) if total else 0,
        }

    # Persona utilization: 有多少样本的 agent 调用了 get_user_profile
    valid_results = [r for r in eval_results if not r.get("runtime_error")]
    profile_called = sum(1 for r in valid_results if r.get("get_profile_calls", 0) > 0)
    n_valid = len(valid_results) or 1

    # Clarification efficiency
    ask_user_counts = [int(r.get("ask_user_calls", 0) or 0) for r in eval_results]
    total_ask_user = sum(ask_user_counts)

    # 如果有 benchmark 数据，计算 slot 触发率
    slot_trigger_stats = _compute_slot_trigger_stats(eval_results, benchmarks) if benchmarks else {}

    base_metrics["by_source"] = by_source
    base_metrics["persona_utilization"] = {
        "profile_called_count": profile_called,
        "profile_called_rate": round(profile_called / n_valid, 4),
    }
    base_metrics["clarification_efficiency"] = {
        "total_ask_user_calls": total_ask_user,
        "avg_ask_user_calls": round(total_ask_user / n_valid, 4),
        "median_ask_user_calls": median(ask_user_counts) if ask_user_counts else 0,
        **slot_trigger_stats,
    }

    return base_metrics


def _compute_slot_trigger_stats(
    eval_results: list[dict],
    benchmarks: dict,
) -> dict:
    """统计 clarification slots 的触发情况。"""
    total_slots = 0
    for r in eval_results:
        sid = r.get("id", "")
        benchmark = benchmarks.get(sid)
        if not benchmark:
            continue
        script = benchmark.get("clarification_script", {})
        slots = script.get("clarification_slots", [])
        total_slots += len(slots)

    return {
        "total_clarification_slots": total_slots,
    }
