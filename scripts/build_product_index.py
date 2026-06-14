"""将 JSONL 商品数据构建为 SQLite + FTS5 索引。

支持两种模式：
  1. 构建商品索引：--input <products_dir> --output <db_path>
  2. 构建评论索引：--reviews-input <reviews_dir> --output <db_path>（需先构建商品索引）
"""
# uv run python3 scripts/build_product_index.py --input data/amazon_raw/products --output data/products/product.db
# uv run python3 scripts/build_product_index.py --reviews-input data/amazon_raw/reviews --output data/products/product.db

import argparse
import json
import logging
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.config import IMPLICIT_ELIGIBLE_FIELDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMPLICIT_ELIGIBLE_DETAIL_KEYS = {
    f.split(".", 1)[1] for f in IMPLICIT_ELIGIBLE_FIELDS
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    store      TEXT NOT NULL DEFAULT '',
    brand      TEXT NOT NULL DEFAULT '',
    price      REAL,
    average_rating    REAL,
    rating_number     INTEGER NOT NULL DEFAULT 0,
    main_category     TEXT NOT NULL DEFAULT '',
    detail_field_count      INTEGER NOT NULL DEFAULT 0,
    implicit_eligible_count INTEGER NOT NULL DEFAULT 0,
    has_features    INTEGER NOT NULL DEFAULT 0,
    has_description INTEGER NOT NULL DEFAULT 0,
    has_price       INTEGER NOT NULL DEFAULT 0,
    data TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
    title,
    content='products'
);

CREATE INDEX IF NOT EXISTS idx_store    ON products(store);
CREATE INDEX IF NOT EXISTS idx_brand    ON products(brand COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_category ON products(main_category);
"""

def _implicit_eligible_count(details: dict) -> int:
    return sum(1 for k in IMPLICIT_ELIGIBLE_DETAIL_KEYS if details.get(k))


def _collect_jsonl_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {p}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {p}")


def build(input_path: str, output_path: str, batch_size: int = 5000) -> None:
    files = _collect_jsonl_files(input_path)
    logger.info("Input files: %s", [f.name for f in files])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(out) + suffix)
        if p.exists():
            p.unlink()

    conn = sqlite3.connect(str(out))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)

    total = 0
    seen_ids: set[str] = set()

    for fpath in files:
        logger.info("Processing %s ...", fpath.name)
        product_rows = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = p.pop("parent_asin")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                p["product_id"] = pid
                p.pop("bought_together", None)
                details = p.get("details", {})
                price = p.get("price")

                product_rows.append((
                    pid,
                    p.get("title") or "",
                    p.get("store") or "",
                    details.get("Brand") or "",
                    price,
                    p.get("average_rating"),
                    p.get("rating_number") or 0,
                    p.get("main_category") or "",
                    len(details),
                    _implicit_eligible_count(details),
                    int(bool(p.get("features"))),
                    int(bool(p.get("description"))),
                    int(price is not None),
                    json.dumps(p, ensure_ascii=False),
                ))
                if len(product_rows) >= batch_size:
                    _flush(conn, product_rows)
                    total += len(product_rows)
                    logger.info("  ... %d products", total)
                    product_rows.clear()

        if product_rows:
            _flush(conn, product_rows)
            total += len(product_rows)

    logger.info("Building products title FTS index...")
    conn.execute("INSERT INTO products_fts(products_fts) VALUES('rebuild')")
    logger.info("Optimizing FTS indexes...")
    conn.execute("INSERT INTO products_fts(products_fts) VALUES('optimize')")
    conn.commit()
    conn.close()
    logger.info("Done. %d products written to %s", total, out)


def _flush(conn: sqlite3.Connection,
           product_rows: list[tuple]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        product_rows,
    )
    conn.commit()


_REVIEWS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id        TEXT NOT NULL,
    rating            REAL,
    title             TEXT NOT NULL DEFAULT '',
    text              TEXT NOT NULL DEFAULT '',
    helpful_vote      INTEGER NOT NULL DEFAULT 0,
    verified_purchase INTEGER NOT NULL DEFAULT 0,
    timestamp         INTEGER
);

CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);

CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts USING fts5(
    title,
    text,
    content='reviews',
    content_rowid='review_id'
);
"""


def _load_product_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT product_id FROM products").fetchall()
    return {r[0] for r in rows}


def _flush_reviews(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    conn.executemany(
        "INSERT INTO reviews(product_id, rating, title, text, helpful_vote, verified_purchase, timestamp) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def build_reviews(db_path: str, reviews_input: str,
                  max_per_product: int = 20, seed: int = 42,
                  batch_size: int = 10000) -> None:
    """将 review JSONL 数据导入已有的商品数据库，每个商品最多保留 max_per_product 条。

    采用两遍扫描法，峰值内存约 1-2 GB。
    """
    files = _collect_jsonl_files(reviews_input)
    logger.info("Review files: %s", [f.name for f in files])

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    product_ids = _load_product_ids(conn)
    logger.info("Loaded %d product_ids from existing DB", len(product_ids))

    conn.execute("DROP TABLE IF EXISTS reviews_fts")
    conn.executescript(_REVIEWS_SCHEMA_SQL)

    total_inserted = 0

    for fpath in files:
        logger.info("Processing reviews: %s ...", fpath.name)

        # ── 遍历1: 统计每个 product_id 的 review 总数 ──
        review_counts: dict[str, int] = {}
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                pid = json.loads(line).get("parent_asin", "")
                if pid in product_ids:
                    review_counts[pid] = review_counts.get(pid, 0) + 1

        logger.info("  Pass 1 done: %d products with reviews", len(review_counts))

        # ── 中间步骤: 对 count > max 的 product 预算选中索引 ──
        selected_indices: dict[str, set[int]] = {}
        for pid, count in review_counts.items():
            if count > max_per_product:
                # 用字符串 seed 播种（Python 对 str 走 SHA-512，不受 PYTHONHASHSEED 影响），
                # 保证每个商品保留的 max_per_product 条评论在跨运行/跨机器时确定可复现。
                prng = random.Random(f"{pid}_{seed}")
                selected_indices[pid] = set(prng.sample(range(count), max_per_product))

        # ── 遍历2: 流式读取并按预算选择 ──
        counters: dict[str, int] = {}
        batch: list[tuple] = []

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pid = r.get("parent_asin", "")
                if pid not in review_counts:
                    continue

                idx = counters.get(pid, 0)
                counters[pid] = idx + 1

                total_count = review_counts[pid]
                if total_count > max_per_product and idx not in selected_indices.get(pid, set()):
                    continue

                batch.append((
                    pid,
                    r.get("rating"),
                    r.get("title", ""),
                    r.get("text", ""),
                    r.get("helpful_vote", 0),
                    int(bool(r.get("verified_purchase"))),
                    r.get("timestamp"),
                ))

                if len(batch) >= batch_size:
                    _flush_reviews(conn, batch)
                    total_inserted += len(batch)
                    logger.info("  ... %d reviews inserted", total_inserted)
                    batch.clear()

        if batch:
            _flush_reviews(conn, batch)
            total_inserted += len(batch)

        del review_counts, selected_indices, counters
        logger.info("  File done. Total so far: %d reviews", total_inserted)

    # ── 构建 FTS 索引 ──
    logger.info("Building reviews FTS index...")
    conn.execute("INSERT INTO reviews_fts(reviews_fts) VALUES('rebuild')")
    conn.execute("INSERT INTO reviews_fts(reviews_fts) VALUES('optimize')")

    # ── 更新 products 表的 review_count 列 ──
    logger.info("Updating review_count on products table...")
    try:
        conn.execute("ALTER TABLE products ADD COLUMN review_count INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # 列已存在
    conn.execute(
        "UPDATE products SET review_count = "
        "(SELECT COUNT(*) FROM reviews WHERE reviews.product_id = products.product_id)"
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    products_with_reviews = conn.execute(
        "SELECT COUNT(*) FROM products WHERE review_count > 0"
    ).fetchone()[0]
    conn.close()
    logger.info(
        "Done. %d reviews written (%d products with reviews) -> %s",
        count, products_with_reviews, db_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SQLite product index from JSONL files")
    parser.add_argument("--input", default=None, help="JSONL file or directory containing product *.jsonl")
    parser.add_argument("--reviews-input", default=None, help="JSONL file or directory containing review *.jsonl")
    parser.add_argument("--output", required=True, help="Output .db file path")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-reviews-per-product", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input:
        build(args.input, args.output, args.batch_size)
    if args.reviews_input:
        build_reviews(args.output, args.reviews_input,
                      max_per_product=args.max_reviews_per_product,
                      seed=args.seed, batch_size=args.batch_size)
