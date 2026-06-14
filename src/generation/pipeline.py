"""数据生成 Pipeline：sample → feature → partition → query/persona/clarification → validate。"""

import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.database.product_db import ProductDB
from src.evaluation.judge import GenerationJudge
from src.generation.config import (
    requires_review_opinions,
    requires_voucher,
)
from src.generation.planning.feature_extractor import extract_features
from src.generation.planning.partition import partition_features
from src.generation.planning.review_opinion_extractor import extract_review_opinions
from src.generation.planning.sampler import ProductSampler
from src.generation.planning.voucher_generator import generate_voucher
from src.generation.composition.clarification_builder import build_clarification_script
from src.generation.composition.persona_builder import build_persona
from src.generation.composition.query_builder import QueryBuilder
from src.generation.composition.rubric_builder import (
    build_rubrics,
    filter_rubrics_by_source,
    merge_rubrics,
)
from src.generation.validation.sample_validator import final_validate_sample
from src.llm import LLMClient, resolve_llm_config

logger = logging.getLogger(__name__)

def load_intents(path: str = "src/intents/catalog.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str = "configs/settings.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean_product(product: dict) -> dict:
    return {
        "product_id": product["product_id"],
        "title": product.get("title", ""),
        "main_category": product.get("main_category", ""),
        "store": product.get("store", ""),
        "average_rating": product.get("average_rating"),
        "rating_number": product.get("rating_number"),
        "price": product.get("price"),
        "features": product.get("features", []),
        "description": product.get("description", []),
        "details": product.get("details", {}),
    }


def _build_one_sample(
    task: dict,
    query_builder: QueryBuilder,
    llm_client: LLMClient,
    generation_judge: GenerationJudge,
    gen_config: dict,
    db: ProductDB | None = None,
) -> dict | None:
    """单样本生成：partition → persona/clarification/query → validate。"""
    intent = task["intent"]
    features = task["features"]
    product = task["product"]
    voucher = task["voucher"]
    sid = task["sid"]
    rng = random.Random(hash(sid) & 0xFFFFFFFF)

    partition_config = gen_config.get("partition", {})
    clarification_config = gen_config.get("clarification", {})



    for attempt in range(3):
        partitioned = partition_features(
        features=features,
        intent=intent,
        product=product,
        persona_ratio=partition_config.get("persona_ratio", 0.25),
        clarification_ratio=partition_config.get("clarification_ratio", 0.25),
        min_query=partition_config.get("min_query_features", 3),
        min_persona=partition_config.get("min_persona_features", 1),
        min_clarification=partition_config.get("min_clarification_features", 1),
        rng=rng,
        )
        if not partitioned:
            if attempt == 2:
                logger.warning("Skipping %s: feature partition failed", sid)
                return None
            else:
                continue

        all_rubrics = build_rubrics(
            partitioned_features=partitioned,
            intent_id=intent["id"],
            product=product,
            voucher=voucher,
        )
        if not all_rubrics:
            if attempt == 2:
                logger.warning("Skipping %s: no rubrics built", sid)
                return None
            else:
                continue

        query_rubrics = filter_rubrics_by_source(all_rubrics, "query")
        persona_rubrics = filter_rubrics_by_source(all_rubrics, "persona")
        clarification_rubrics = filter_rubrics_by_source(all_rubrics, "clarification")

        user_persona = build_persona(
            persona_rubrics=persona_rubrics,
            product=product,
            llm_client=llm_client,
            rng=rng,
        )

        clarification_script = build_clarification_script(
            clarification_rubrics=clarification_rubrics,
            product=product,
            llm_client=llm_client,
            max_turns=clarification_config.get("max_turns", 5),
        )

        validation_product = product
        if requires_review_opinions(intent["id"]) and db:
            enriched = dict(product)
            enriched["reviews"] = db.get_reviews(product["product_id"], limit=20)
            validation_product = enriched

        sample = {
            "id": sid,
            "intent_category": intent["id"],
            "user_query": "",
            "user_persona": user_persona,
            "clarification_script": clarification_script,
            "target_product": _clean_product(product),
            "rubrics": all_rubrics,
            "implicit_rubric_ids": [],
            "difficulty_metrics": {"implicit_count": 0},
            "generation_artifacts": {"sample_retry_count": 0, "judge_issues": []},
            "extracted_features": [
                {"field": f["field"], "value": f["value"], "type": f["type"]}
                for f in features
            ],
            "feature_partition": {
                "query_count": len(partitioned["query"]),
                "persona_count": len(partitioned["persona"]),
                "clarification_count": len(partitioned["clarification"]),
            },
            "voucher": voucher,
            "validation_passed": None,
            "validation_issues": [],
            "judge_passed": None,
            "judge_issues": [],
        }

        excluded_values = []
        for r in persona_rubrics + clarification_rubrics:
            ev = str(r.get("expected_value", "")).strip()
            if ev and r.get("type") not in ("negative_attribute", "numeric_range"):
                excluded_values.append(ev)
        max_rubric_id = max(
            (int(r["id"][1:]) for r in all_rubrics if r["id"].startswith("r") and r["id"][1:].isdigit()),
            default=0,
        )
        try:
            query_result = query_builder.build(
                intent, product, query_rubrics, voucher,
                excluded_values=excluded_values,
                max_rubric_id=max_rubric_id,
            )
        except Exception as e:
            logger.error("LLM call failed for %s: %s", sid, e)
            sample["sample_status"] = "generation_runtime_error"
            sample["error_message"] = str(e)
            return sample

        sample["user_query"] = query_result["user_query"]
        sample["implicit_rubric_ids"] = query_result.get("implicit_rubric_ids", [])
        sample["difficulty_metrics"]["implicit_count"] = len(sample["implicit_rubric_ids"])

        compiled_query_rubrics = query_result["rubrics"]
        merged_rubrics = merge_rubrics(all_rubrics, compiled_query_rubrics)
        sample["rubrics"] = merged_rubrics

        try:
            validation = final_validate_sample(
                user_query=query_result["user_query"],
                rubrics_scaffold=query_rubrics,
                rubrics=compiled_query_rubrics,
                product=validation_product,
                voucher=voucher,
                implicit_rubric_ids=sample["implicit_rubric_ids"],
                min_implicit_count=query_result.get("min_implicit_count", 0),
                generation_judge=generation_judge,
                user_persona=user_persona,
                clarification_script=clarification_script,
                persona_rubrics=persona_rubrics,
                clarification_rubrics=clarification_rubrics,
                all_rubrics=merged_rubrics,
            )
        except Exception as e:
            logger.error("Validation failed for %s: %s", sid, e)
            if attempt < 2:
                logger.warning("Retrying %s after validation error", sid)
                continue
            sample["sample_status"] = "validation_runtime_error"
            sample["error_message"] = str(e)
            return sample

        sample["generation_artifacts"]["sample_retry_count"] = attempt
        sample["generation_artifacts"]["judge_issues"] = validation.get("judge_issues", [])
        sample["validation_passed"] = validation["passed"]
        sample["validation_issues"] = validation["issues"]
        sample["judge_passed"] = validation["passed"]
        sample["judge_issues"] = validation.get("judge_issues", [])

        if validation["passed"]:
            sample["sample_status"] = "ok"
            logger.info("Generated %s (%s)", sid, intent["id"])
            return sample

        if attempt < 2:
            logger.warning(
                "Retrying %s -> %s", sid, "; ".join(validation["issues"][:6]),
            )
            continue

        sample["sample_status"] = "final_validation_failed"
        logger.warning(
            "Recording %s as final_validation_failed -> %s",
            sid, "; ".join(validation["issues"][:8]),
        )
        return sample


def _prepare_task(
    intent: dict, product: dict, sid: str,
    sampler: ProductSampler, rng: random.Random,
    min_features: int, max_features: int,
    db: ProductDB | None = None,
) -> dict | None:
    underused = sampler.get_underused_features()
    features = extract_features(
        product, intent["id"],
        min_features=min_features,
        max_features=max_features,
        rng=rng,
        prefer_fields=underused,
    )

    if len(features) < min_features:
        logger.warning("Skipping %s: only %d features extracted", sid, len(features))
        return None

    voucher = None
    if requires_voucher(intent["id"]):
        voucher = generate_voucher(product, rng)
        if not voucher:
            logger.warning("Skipping %s: failed to generate voucher", sid)
            return None

    is_review_driven = requires_review_opinions(intent["id"])
    pending_reviews = None
    if is_review_driven and db:
        reviews = db.get_reviews(product["product_id"], limit=20)
        if len(reviews) < 10:
            logger.warning("Skipping %s: insufficient reviews (%d)", sid, len(reviews))
            return None
        pending_reviews = reviews

    sampler.update_feature_usage([f["field"].split(".")[-1] for f in features])
    task = {
        "sid": sid,
        "intent": intent,
        "product": product,
        "features": features,
        "voucher": voucher,
    }
    if pending_reviews is not None:
        task["_pending_reviews"] = pending_reviews
    return task


def _complete_review_task(task: dict, llm_client: LLMClient,
                          rng: random.Random) -> dict | None:
    sid = task["sid"]
    product = task["product"]
    opinions = extract_review_opinions(
        reviews=task["_pending_reviews"],
        product=product,
        llm_client=llm_client, n_opinions=2, rng=rng,
    )
    if len(opinions) < 2:
        logger.warning("Skipping %s: failed to extract review opinions", sid)
        return None

    for opinion in opinions:
        task["features"].append({
            "field": "review",
            "value": opinion["opinion"],
            "type": "review_opinion",
            "product_id": product["product_id"],
            "rubric": f"A review of the recommended product should mention: {opinion['opinion']}",
            "_opinion_data": opinion,
        })

    del task["_pending_reviews"]
    return task


def _build_task_pool_for_intent(
    intent: dict, target_count: int,
    db: ProductDB, sampler: ProductSampler,
    rng: random.Random, min_features: int,
    max_features: int, start_sample_id: int,
) -> tuple[list[dict], int]:
    tasks = []
    seen_ids: set[str] = set()
    sample_id = start_sample_id
    batch_size = max(target_count * 10, 200)
    max_rounds = 10

    for round_idx in range(max_rounds):
        if len(tasks) >= target_count:
            break

        products = sampler.sample_batch(intent, batch_size=batch_size)
        if not products:
            logger.warning("Intent %s round %d: no candidates", intent["id"], round_idx)
            continue

        hit = 0
        for product in products:
            key = product["product_id"]
            if key in seen_ids:
                continue

            sid = f"v3_sample_{sample_id + 1:04d}"
            task = _prepare_task(
                intent=intent, product=product, sid=sid,
                sampler=sampler, rng=rng,
                min_features=min_features, max_features=max_features,
                db=db,
            )
            if not task:
                continue

            seen_ids.add(key)
            tasks.append(task)
            sample_id += 1
            hit += 1
            if len(tasks) >= target_count:
                break

        logger.info(
            "Intent %s round %d: %d/%d hit, %d/%d collected",
            intent["id"], round_idx, hit, len(products), len(tasks), target_count,
        )

    if len(tasks) < target_count:
        logger.warning(
            "Intent %s only prepared %d/%d valid tasks",
            intent["id"], len(tasks), target_count,
        )
    return tasks, sample_id


def run_pipeline(config_path: str = "configs/settings.yaml",
                 output_path: str | None = None,
                 intent_filter: list[str] | None = None,
                 samples_per_intent: int | None = None,
                 num_threads: int | None = None):
    load_dotenv()
    config = load_config(config_path)
    gen_config = config["generation"]

    seed = gen_config.get("seed", 42)
    intent_samples_config = gen_config.get("intents", {})
    default_samples = gen_config.get("samples_per_intent", 40)
    min_features = gen_config.get("min_features", 8)
    max_features = gen_config.get("max_features", 20)
    threads = num_threads or gen_config.get("num_threads", config.get("num_threads", 16))
    min_category_size = gen_config.get("sampling", {}).get("min_category_size", 30)

    if not output_path:
        output_dir = config["data"]["benchmark_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, "benchmark.jsonl")

    logger.info("Loading product database...")
    db = ProductDB(config["data"]["db_path"])

    intents = load_intents()
    if intent_filter:
        intents = [i for i in intents if i["id"] in intent_filter]
    logger.info("Generating for %d intent categories", len(intents))

    llm_config = resolve_llm_config(config, section="generation")
    llm_client = LLMClient(mode=llm_config.get("provider", "openai"), config=llm_config)
    query_builder = QueryBuilder(llm_client=llm_client, seed=seed)
    generation_judge = GenerationJudge(llm_client=llm_client)

    sampler = ProductSampler(db, seed=seed, min_category_size=min_category_size)
    rng = random.Random(seed)

    tasks = []
    sample_id = 0

    for intent in intents:
        n_samples = samples_per_intent or intent_samples_config.get(intent["id"], default_samples)
        effective_target = int(n_samples * 1.3) if requires_review_opinions(intent["id"]) else n_samples
        logger.info(
            "Building task pool for intent: %s (%d samples, collect %d)",
            intent["id"], n_samples, effective_target,
        )
        intent_tasks, sample_id = _build_task_pool_for_intent(
            intent=intent, target_count=effective_target,
            db=db, sampler=sampler, rng=rng,
            min_features=min_features, max_features=max_features,
            start_sample_id=sample_id,
        )
        tasks.extend(intent_tasks)

    pending = [t for t in tasks if "_pending_reviews" in t]
    if pending:
        logger.info("Extracting review opinions for %d tasks...", len(pending))
        task_seeds = {t["sid"]: rng.randint(0, 2**32 - 1) for t in pending}
        completed_sids: set[str] = set()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(
                    _complete_review_task, t, llm_client,
                    random.Random(task_seeds[t["sid"]]),
                ): t["sid"]
                for t in pending
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    completed_sids.add(result["sid"])
        failed_sids = {t["sid"] for t in pending} - completed_sids
        if failed_sids:
            tasks = [t for t in tasks if t["sid"] not in failed_sids]
            logger.info("Removed %d tasks after review opinion extraction failure", len(failed_sids))

    logger.info("Prepared %d tasks, generating with %d threads...", len(tasks), threads)

    all_samples = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(
                _build_one_sample, t, query_builder, llm_client,
                generation_judge, gen_config, db,
            ): t["sid"]
            for t in tasks
        }
        for future in as_completed(futures):
            sid = futures[future]
            sample = future.result()
            if sample:
                all_samples.append(sample)

    all_samples.sort(key=lambda s: s["id"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("Generated %d samples -> %s", len(all_samples), output_path)
    token_usage = llm_client.get_token_usage()
    logger.info(
        "Total LLM tokens: input=%d thinking=%d output=%d",
        token_usage["input_tokens"],
        token_usage["thinking_tokens"],
        token_usage["output_tokens"],
    )
    return all_samples


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--intents", nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(args.config, args.output, args.intents, args.samples, args.threads)


if __name__ == "__main__":
    main()
