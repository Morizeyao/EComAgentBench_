"""Sample 校验：含 persona/clarification 安全审查。"""

from src.generation.utils import normalize_text
from src.generation.composition.persona_builder import validate_persona
from src.generation.composition.clarification_builder import validate_clarification_script


def validate_generation_output(
    *,
    user_query: str,
    rubrics_scaffold: list[dict],
    rubrics: list[dict],
    product: dict,
    implicit_rubric_ids: list[str],
    min_implicit_count: int,
    user_persona: dict | None = None,
    clarification_script: dict | None = None,
    persona_rubrics: list[dict] | None = None,
    clarification_rubrics: list[dict] | None = None,
) -> dict:
    """验证：基础对齐 + persona 安全 + clarification 完备 + 信息不泄漏。"""
    issues = []

    # 1. Query-source rubrics 的基础验证
    issues.extend(_collect_alignment_issues(
        user_query=user_query,
        scaffold=rubrics_scaffold,
        rubrics=rubrics,
        product=product,
    ))
    issues.extend(_collect_implicit_selection_issues(
        rubrics_scaffold=rubrics_scaffold,
        implicit_rubric_ids=implicit_rubric_ids,
        min_implicit_count=min_implicit_count,
    ))
    issues.extend(_collect_implicit_leaks(
        user_query=user_query,
        rubrics=rubrics,
        implicit_rubric_ids=implicit_rubric_ids,
    ))

    # 2. Persona 安全审查
    if user_persona and persona_rubrics:
        persona_issues = validate_persona(user_persona, persona_rubrics)
        issues.extend(persona_issues)

    # 3. Clarification 完备性
    if clarification_script and clarification_rubrics:
        cl_issues = validate_clarification_script(clarification_script, clarification_rubrics)
        issues.extend(cl_issues)

    return {
        "passed": not issues,
        "issues": issues[:15],
    }


def final_validate_sample(
    *,
    user_query: str,
    rubrics_scaffold: list[dict],
    rubrics: list[dict],
    product: dict,
    voucher: dict | None,
    implicit_rubric_ids: list[str],
    min_implicit_count: int,
    generation_judge,
    user_persona: dict | None = None,
    clarification_script: dict | None = None,
    persona_rubrics: list[dict] | None = None,
    clarification_rubrics: list[dict] | None = None,
    all_rubrics: list[dict] | None = None,
) -> dict:
    """完整验证：硬门槛 + 三源 judge。"""
    gate_result = validate_generation_output(
        user_query=user_query,
        rubrics_scaffold=rubrics_scaffold,
        rubrics=rubrics,
        product=product,
        implicit_rubric_ids=implicit_rubric_ids,
        min_implicit_count=min_implicit_count,
        user_persona=user_persona,
        clarification_script=clarification_script,
        persona_rubrics=persona_rubrics,
        clarification_rubrics=clarification_rubrics,
    )
    if not gate_result["passed"]:
        return {
            "passed": False,
            "failed_stage": "hard_gate",
            "issues": gate_result["issues"],
            "judge_issues": [],
            "rubric_results": [],
        }

    judge_rubrics = all_rubrics or rubrics
    judge_result = generation_judge.judge_generation_sample(
        user_query=user_query,
        user_persona=user_persona,
        clarification_script=clarification_script,
        product=product,
        rubrics=judge_rubrics,
        voucher=voucher,
        implicit_rubric_ids=implicit_rubric_ids,
    )
    issues = list(judge_result.get("issues", []))
    passed = bool(judge_result.get("passed"))
    return {
        "passed": passed,
        "failed_stage": None if passed else "judge",
        "issues": issues,
        "judge_issues": issues,
        "rubric_results": list(judge_result.get("rubric_results", [])),
    }


def _collect_alignment_issues(
    *,
    user_query: str,
    scaffold: list[dict],
    rubrics: list[dict],
    product: dict,
) -> list[str]:
    issues: list[str] = []
    query_norm = normalize_text(user_query)

    compiled_index = {rubric.get("id"): rubric for rubric in rubrics}
    for scaffold_rubric in scaffold:
        compiled_rubric = compiled_index.get(scaffold_rubric.get("id"))
        if not compiled_rubric:
            issues.append(f"missing rubric {scaffold_rubric.get('id')}")
            continue

        query_surface = str(compiled_rubric.get("query_surface", "")).strip()
        if not query_surface:
            issues.append(f"{scaffold_rubric.get('id')}: missing query_surface")
            continue

        if normalize_text(query_surface) not in query_norm:
            issues.append(f"{scaffold_rubric.get('id')}: query_surface is not anchored in the final query")

    entity_rubrics = [rubric for rubric in rubrics if rubric.get("type") == "entity_match"]
    if not entity_rubrics:
        issues.append("missing entity_match rubric")
    else:
        for rubric in entity_rubrics:
            query_surface = str(rubric.get("query_surface", "")).strip()
            title_span = str(rubric.get("title_span", "")).strip()
            if not query_surface:
                issues.append(f"{rubric.get('id')}: entity rubric missing query_surface")
            if not title_span:
                issues.append(f"{rubric.get('id')}: entity rubric missing title_span")
                continue
            title_norm = normalize_text(product.get("title", ""))
            if normalize_text(title_span) not in title_norm:
                issues.append(f"{rubric.get('id')}: entity rubric title_span is not grounded in the target title")

    return issues[:12]


def _collect_implicit_selection_issues(
    *,
    rubrics_scaffold: list[dict],
    implicit_rubric_ids: list[str],
    min_implicit_count: int,
) -> list[str]:
    eligible_ids = {
        str(rubric.get("id"))
        for rubric in rubrics_scaffold
        if rubric.get("implicit_eligible")
    }
    invalid_ids = [rubric_id for rubric_id in implicit_rubric_ids if rubric_id not in eligible_ids]
    issues: list[str] = []
    if invalid_ids:
        issues.append(
            "implicit_rubric_ids contains rubric ids outside the eligible pool: "
            + ", ".join(invalid_ids[:10])
        )
    if len(implicit_rubric_ids) < min_implicit_count:
        issues.append(
            f"implicit_rubric_ids only selected {len(implicit_rubric_ids)} rubrics, below required minimum {min_implicit_count}"
        )
    return issues[:10]


def _collect_implicit_leaks(
    *,
    user_query: str,
    rubrics: list[dict],
    implicit_rubric_ids: list[str],
) -> list[str]:
    query_norm = normalize_text(user_query or "")
    implicit_id_set = {str(rubric_id) for rubric_id in implicit_rubric_ids}
    issues: list[str] = []
    for rubric in rubrics:
        if str(rubric.get("id")) not in implicit_id_set:
            continue
        expected_value = str(rubric.get("expected_value", "")).strip()
        expected_norm = normalize_text(expected_value)
        if expected_norm and len(expected_norm) > 2 and expected_norm in query_norm:
            issues.append(
                f"{rubric.get('id')}: implicit rubric leaks expected value '{expected_value}'"
            )
    return issues[:10]
