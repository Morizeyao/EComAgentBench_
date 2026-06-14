"""从匿名 HuggingFace dataset 下载预构建的 product.db（开箱即用，免重建）。

数据集作者已授权发布。product.db（~25GB）托管在 HF dataset，benchmark.jsonl 随代码仓库。
若不想下载完整 db，可改用 scripts/download_amazon_reviews.py + scripts/build_product_index.py 重建。

用法：
  # 默认下载（已内置 HF dataset repo）
  uv run python scripts/download_db.py
  # 或显式指定 / 用环境变量覆盖 repo
  uv run python scripts/download_db.py --repo ecomagentbench/EcomAgentBenchProductDB
  HF_DATASET_REPO=ecomagentbench/EcomAgentBenchProductDB uv run python scripts/download_db.py

若 .env 提供 HF_TOKEN，会自动用于鉴权/提速。
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# 加载 .env（若存在）：huggingface_hub 会自动读取 HF_TOKEN。
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 预构建 product.db 托管在该 HF dataset 根目录；可用 --repo / 环境变量覆盖。
DEFAULT_REPO = os.getenv("HF_DATASET_REPO", "ecomagentbench/EcomAgentBenchProductDB")
DB_FILENAME = "product.db"


def download_db(repo: str, output_path: str) -> None:
    out = Path(output_path)
    if out.exists():
        logger.info("已存在，跳过：%s（如需重新下载请先删除该文件）", out)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("从 HF dataset %s 下载 %s（~25GB，请耐心等待）...", repo, DB_FILENAME)
    local = hf_hub_download(repo_id=repo, filename=DB_FILENAME, repo_type="dataset")
    logger.info("复制到 %s ...", out)
    shutil.copyfile(local, out)
    logger.info("完成：%s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the prebuilt product.db from a Hugging Face dataset."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id（包含 product.db）")
    parser.add_argument("--output", default="data/products/product.db")
    args = parser.parse_args()

    download_db(args.repo, args.output)
