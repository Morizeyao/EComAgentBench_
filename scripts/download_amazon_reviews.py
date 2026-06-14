"""从 HuggingFace 下载 Amazon Reviews 2023 原始数据，导出为 build_product_index.py 可消费的 JSONL。

数据来源：McAuley-Lab/Amazon-Reviews-2023
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
仅供研究用途，使用须遵循其原始条款。

该脚本直接下载数据集仓库内的原始 JSONL（字段为原生 JSON，details 为真正的 dict），
无需字段转换即可被 scripts/build_product_index.py 读取：
  - 商品元数据：raw/meta_categories/meta_<Category>.jsonl
  - 评论：       raw/review_categories/<Category>.jsonl

论文构建 product.db 时使用了其中四个类目的【全量】数据（见 PAPER_CATEGORIES），
合计约 372 万商品 / 2135 万评论，已覆盖 benchmark 中全部目标商品。用 --paper 一键下载。

若 .env 中提供了 HF_TOKEN，会自动用于提高下载限速（匿名请求限速较低）。

用法：
  # 1) 论文所用的四个类目全量（推荐，严格复现，体积达数十 GB）
  uv run python scripts/download_amazon_reviews.py --paper --output-dir data/amazon_raw

  # 2) 单个类目（快速试跑，几十 MB）
  uv run python scripts/download_amazon_reviews.py --categories All_Beauty --output-dir data/amazon_raw

  # 3) 全部类目
  uv run python scripts/download_amazon_reviews.py --all --output-dir data/amazon_raw

下载完成后构建数据库：
  uv run python scripts/build_product_index.py --input data/amazon_raw/products --output data/products/product.db
  uv run python scripts/build_product_index.py --reviews-input data/amazon_raw/reviews --output data/products/product.db \
      --max-reviews-per-product 20 --seed 42
"""

import argparse
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# 加载 .env（若存在）：huggingface_hub 会自动读取环境变量 HF_TOKEN。
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"

# 论文构建 product.db 实际使用的四个类目（全量）。
PAPER_CATEGORIES = [
    "All_Beauty",
    "Electronics",
    "Cell_Phones_and_Accessories",
    "Office_Products",
]

# Amazon Reviews 2023 的全部商品类目（--all 会拉取全部）。
ALL_CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
    "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
    "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement",
    "Toys_and_Games", "Video_Games", "Unknown",
]


def _download(remote_file: str, dest: Path) -> None:
    """从 HF Hub 下载单个 JSONL 并放到目标位置（已存在则跳过）。"""
    if dest.exists():
        logger.info("已存在，跳过：%s", dest)
        return
    logger.info("下载 %s ...", remote_file)
    local = hf_hub_download(repo_id=HF_REPO, filename=remote_file, repo_type="dataset")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # HF cache 是只读 blob 软链，复制成独立文件供 build_product_index.py 直接读取。
    shutil.copyfile(local, dest)
    logger.info("已保存到 %s", dest)


def download(categories: list[str], output_dir: str) -> None:
    out = Path(output_dir)
    products_dir = out / "products"
    reviews_dir = out / "reviews"
    logger.info("将下载 %d 个类目：%s", len(categories), ", ".join(categories))
    for cat in categories:
        _download(f"raw/meta_categories/meta_{cat}.jsonl", products_dir / f"{cat}.jsonl")
        _download(f"raw/review_categories/{cat}.jsonl", reviews_dir / f"{cat}.jsonl")
    logger.info("完成。products -> %s ，reviews -> %s", products_dir, reviews_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 raw data from the Hugging Face Hub."
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="下载论文所用的四个类目全量（All_Beauty / Electronics / Cell_Phones_and_Accessories / Office_Products）",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="下载全部类目（数据量极大）",
    )
    parser.add_argument(
        "--categories", nargs="+", default=["All_Beauty"],
        help="要下载的类目（默认 All_Beauty，便于快速试跑）",
    )
    parser.add_argument("--output-dir", default="data/amazon_raw")
    args = parser.parse_args()

    if args.all:
        cats = ALL_CATEGORIES
    elif args.paper:
        cats = PAPER_CATEGORIES
    else:
        cats = args.categories
    download(cats, args.output_dir)
