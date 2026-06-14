"""批量预测入口。"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.database.product_db import ProductDB
from src.llm import LLMClient, resolve_prediction_mode_config
from src.prediction.agent import ShoppingAgent

logger = logging.getLogger(__name__)


def _predict_one_sample(sample: dict, agent: ShoppingAgent, model_name: str) -> dict:
    result = agent.run(sample)
    return {
        "id": sample["id"],
        "intent_category": sample["intent_category"],
        "user_query": sample["user_query"],
        "predicted_product_id": result["predicted_product_id"],
        "reasoning": result["reasoning"],
        "trajectory": result["trajectory"],
        "reasoning_trace": result["reasoning_trace"],
        "model": model_name,
        "total_tool_calls": result["total_tool_calls"],
        "ask_user_calls": result["ask_user_calls"],
        "get_profile_calls": result["get_profile_calls"],
        "finished": result["finished"],
    }


def run_prediction(config_path: str = "configs/settings.yaml",
                   mode: str | None = None):
    load_dotenv()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pred_config = config["prediction"]
    mode, _, llm_config = resolve_prediction_mode_config(config, mode)
    threads = pred_config.get("num_threads", config.get("num_threads", 16))
    llm = LLMClient(mode=mode, config=llm_config)

    benchmark_dir = config["data"]["benchmark_dir"]
    benchmark_path = os.path.join(benchmark_dir, "benchmark.jsonl")
    safe_model = llm.model.replace("/", "_")
    output_path = os.path.join(benchmark_dir, f"predictions_{safe_model}.jsonl")

    logger.info("Loading product database...")
    db = ProductDB(config["data"]["db_path"])

    logger.info("Using %s mode, model: %s", mode, llm.model)

    agent = ShoppingAgent(
        db, llm,
        max_steps=pred_config.get("max_steps", 60),
    )

    samples = []
    skipped_invalid = 0
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if sample.get("sample_status", "ok") != "ok":
                skipped_invalid += 1
                continue
            samples.append(sample)
    logger.info(
        "Loaded %d valid samples from %s (skipped %d non-ok samples)",
        len(samples), benchmark_path, skipped_invalid,
    )

    logger.info("Running predictions with %d threads...", threads)
    all_predictions = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(_predict_one_sample, s, agent, llm.model): s["id"]
            for s in samples
        }
        sample_index = {s["id"]: s for s in samples}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                prediction = future.result()
                all_predictions.append(prediction)
                logger.info("[%d/%d] Completed %s", len(all_predictions), len(samples), sid)
            except Exception as exc:
                logger.exception("Failed to predict %s", sid)
                s = sample_index[sid]
                all_predictions.append({
                    "id": sid,
                    "intent_category": s.get("intent_category", ""),
                    "user_query": s.get("user_query", ""),
                    "predicted_product_id": None,
                    "reasoning": "",
                    "trajectory": [],
                    "reasoning_trace": [],
                    "model": llm.model,
                    "total_tool_calls": 0,
                    "ask_user_calls": 0,
                    "get_profile_calls": 0,
                    "finished": False,
                    "error": str(exc),
                })

    all_predictions.sort(key=lambda p: p["id"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for p in all_predictions:
            f_out.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Predictions saved to %s", output_path)
    token_usage = llm.get_token_usage()
    logger.info(
        "Total LLM tokens: input=%d thinking=%d output=%d",
        token_usage["input_tokens"],
        token_usage["thinking_tokens"],
        token_usage["output_tokens"],
    )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--mode", choices=["openai_api", "openrouter", "gemini", "claude"], default=None)
    args = parser.parse_args()
    run_prediction(args.config, args.mode)
