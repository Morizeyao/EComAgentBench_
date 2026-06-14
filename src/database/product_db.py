"""商品数据库，基于 SQLite + FTS5 提供查询和采样功能。

使用前需通过 scripts/build_product_index.py 将 JSONL 数据构建为 .db 文件。
"""

import json
import logging
import re
import sqlite3
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

_FTS5_SPECIAL = re.compile(r'["\*\(\)\-\+\^~:]')


class ProductDB:
    """SQLite 商品数据库（线程安全，每线程独立连接）。"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()

        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        logger.info("ProductDB opened: %d products from %s", count, db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA query_only=ON")
            self._local.conn = conn
        return conn

    # ── 核心读取 ──

    def get_product(self, pid: str) -> dict | None:
        row = self._get_conn().execute(
            "SELECT data FROM products WHERE product_id = ?", (pid,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_products(self, pids: list[str]) -> list[dict]:
        if not pids:
            return []
        placeholders = ",".join("?" for _ in pids)
        rows = self._get_conn().execute(
            f"SELECT product_id, data FROM products WHERE product_id IN ({placeholders})",
            pids,
        ).fetchall()
        by_id = {r[0]: json.loads(r[1]) for r in rows}
        return [by_id[pid] for pid in pids if pid in by_id]

    # ── FTS5 搜索 ──

    def search(self, query: str, limit: int = 20, offset: int = 0) -> list[str]:
        """执行 products_fts 的 BM25 搜索，返回 product_id 列表。"""
        tokens = query.strip().split()
        if not tokens:
            return []
        sanitized = []
        for t in tokens:
            clean = _FTS5_SPECIAL.sub(" ", t).strip()
            if clean:
                sanitized.append(f'"{clean}"')
        if not sanitized:
            return []
        fts_query = " ".join(sanitized)
        try:
            rows = self._get_conn().execute(
                "SELECT p.product_id "
                "FROM products_fts "
                "JOIN products p ON p.rowid = products_fts.rowid "
                "WHERE products_fts MATCH ? "
                "ORDER BY bm25(products_fts) "
                "LIMIT ? OFFSET ?",
                (fts_query, limit, offset),
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed on products_fts for: %s", fts_query[:200])
            return []

    # ── 索引查询 ──

    def get_store_products(self, store: str) -> list[str]:
        rows = self._get_conn().execute(
            "SELECT product_id FROM products WHERE store = ?", (store,)
        ).fetchall()
        return [r[0] for r in rows]

    def get_brand_products(self, brand: str) -> list[str]:
        rows = self._get_conn().execute(
            "SELECT product_id FROM products WHERE brand = ? COLLATE NOCASE",
            (brand,),
        ).fetchall()
        return [r[0] for r in rows]

    # ── 采样查询 ──

    def sample_products(
        self,
        n: int = 1,
        *,
        min_detail_fields: int = 0,
        min_rating: float | None = None,
        min_rating_number: int | None = None,
        require_price: bool = False,
        require_content: bool = False,
        min_implicit_eligible: int = 0,
        min_reviews: int | None = None,
        balance_category: bool = False,
        min_category_frac: float = 0.01,
    ) -> list[dict]:
        """随机采样 n 个满足条件的商品，返回完整 product dict。

        balance_category=True 时按品类均衡采样：每个品类的采样数与其在库中的
        占比成正比，占比低于 min_category_frac 的品类被排除。
        """
        clauses, params = self._build_filter_clauses(
            min_detail_fields=min_detail_fields,
            min_rating=min_rating,
            min_rating_number=min_rating_number,
            require_price=require_price,
            require_content=require_content,
            min_implicit_eligible=min_implicit_eligible,
            min_reviews=min_reviews,
        )
        where = " AND ".join(clauses) if clauses else "1=1"

        if not balance_category:
            rows = self._get_conn().execute(
                f"SELECT data FROM products WHERE {where} ORDER BY RANDOM() LIMIT ?",
                (*params, n),
            ).fetchall()
            return [json.loads(r[0]) for r in rows]

        # 按品类均衡采样：先查各品类合格数，按占比分配名额，逐品类随机采样
        conn = self._get_conn()
        cat_rows = conn.execute(
            f"SELECT main_category, COUNT(*) FROM products WHERE {where} "
            f"GROUP BY main_category",
            params,
        ).fetchall()
        total = sum(r[1] for r in cat_rows)
        if total == 0:
            return []
        cats = [(r[0], r[1]) for r in cat_rows if r[1] / total >= min_category_frac]
        cat_total = sum(c[1] for c in cats)

        results = []
        for cat, cnt in cats:
            k = max(1, round(n * cnt / cat_total))
            rows = conn.execute(
                f"SELECT data FROM products WHERE {where} AND main_category = ? "
                f"ORDER BY RANDOM() LIMIT ?",
                (*params, cat, k),
            ).fetchall()
            results.extend(json.loads(r[0]) for r in rows)
        return results

    def get_eligible_stores(
        self,
        min_products: int = 1,
        *,
        min_implicit_eligible: int = 0,
    ) -> list[str]:
        """返回拥有至少 min_products 个合格商品的店铺名列表（结果会缓存）。"""
        return self._eligible_stores_cached(min_products, min_implicit_eligible)

    @lru_cache(maxsize=32)
    def _eligible_stores_cached(self, min_products: int,
                                min_implicit_eligible: int) -> list[str]:
        clauses, params = self._build_filter_clauses(
            min_implicit_eligible=min_implicit_eligible,
        )
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._get_conn().execute(
            f"SELECT store FROM products WHERE {where} AND store != '' "
            f"GROUP BY store HAVING COUNT(*) >= ?",
            (*params, min_products),
        ).fetchall()
        result = [r[0] for r in rows]
        logger.info("get_eligible_stores(min_products=%d, min_implicit=%d): %d stores",
                     min_products, min_implicit_eligible, len(result))
        return result

    def get_store_product_ids(
        self,
        store: str,
        *,
        min_implicit_eligible: int = 0,
    ) -> list[str]:
        """返回指定店铺中满足条件的商品 ID 列表。"""
        clauses, params = self._build_filter_clauses(
            min_implicit_eligible=min_implicit_eligible,
        )
        clauses.append("store = ?")
        params.append(store)
        where = " AND ".join(clauses)
        rows = self._get_conn().execute(
            f"SELECT product_id FROM products WHERE {where}",
            params,
        ).fetchall()
        return [r[0] for r in rows]

    # ── Review 查询 ──

    def get_reviews(self, product_id: str, limit: int = 20) -> list[dict]:
        """获取指定商品的 review 列表。"""
        rows = self._get_conn().execute(
            "SELECT review_id, product_id, rating, title, text, "
            "helpful_vote, verified_purchase, timestamp "
            "FROM reviews WHERE product_id = ? LIMIT ?",
            (product_id, limit),
        ).fetchall()
        return [
            {"review_id": r[0], "product_id": r[1], "rating": r[2],
             "title": r[3], "text": r[4], "helpful_vote": r[5],
             "verified_purchase": bool(r[6]), "timestamp": r[7]}
            for r in rows
        ]

    def search_reviews(self, query: str, product_id: str,
                       limit: int = 5) -> list[dict]:
        """BM25 搜索指定商品的 review，返回完整字段。"""
        tokens = query.strip().split()
        if not tokens:
            return []
        sanitized = []
        for t in tokens:
            clean = _FTS5_SPECIAL.sub(" ", t).strip()
            if clean:
                sanitized.append(f'"{clean}"')
        if not sanitized:
            return []
        fts_query = " ".join(sanitized)
        try:
            rows = self._get_conn().execute(
                "SELECT r.review_id, r.product_id, r.rating, r.title, r.text, "
                "r.helpful_vote, r.verified_purchase, r.timestamp "
                "FROM reviews_fts "
                "JOIN reviews r ON r.review_id = reviews_fts.rowid "
                "WHERE reviews_fts MATCH ? AND r.product_id = ? "
                "ORDER BY bm25(reviews_fts) LIMIT ?",
                (fts_query, product_id, limit),
            ).fetchall()
            return [
                {"review_id": r[0], "product_id": r[1], "rating": r[2],
                 "title": r[3], "text": r[4], "helpful_vote": r[5],
                 "verified_purchase": bool(r[6]), "timestamp": r[7]}
                for r in rows
            ]
        except sqlite3.OperationalError:
            logger.warning("Reviews FTS query failed: %s", fts_query[:200])
            return []

    # ── 预加载采样（仅缓存 ID，节省内存） ──

    def get_eligible_ids_by_category(
        self,
        *,
        min_detail_fields: int = 0,
        min_rating: float | None = None,
        min_rating_number: int | None = None,
        require_price: bool = False,
        require_content: bool = False,
        min_implicit_eligible: int = 0,
        min_reviews: int | None = None,
    ) -> dict[str, list[str]]:
        """返回按品类分组的合格商品 ID（结果会缓存）。"""
        return self._eligible_ids_by_category_cached(
            min_detail_fields, min_rating, min_rating_number,
            require_price, require_content, min_implicit_eligible, min_reviews)

    @lru_cache(maxsize=16)
    def _eligible_ids_by_category_cached(
        self, min_detail_fields, min_rating, min_rating_number,
        require_price, require_content, min_implicit_eligible, min_reviews,
    ) -> dict[str, list[str]]:
        clauses, params = self._build_filter_clauses(
            min_detail_fields=min_detail_fields,
            min_rating=min_rating,
            min_rating_number=min_rating_number,
            require_price=require_price,
            require_content=require_content,
            min_implicit_eligible=min_implicit_eligible,
            min_reviews=min_reviews,
        )
        where = " AND ".join(clauses) if clauses else "1=1"
        cursor = self._get_conn().execute(
            f"SELECT product_id, main_category FROM products WHERE {where}", params)
        by_cat: dict[str, list[str]] = {}
        count = 0
        for pid, cat in cursor:
            by_cat.setdefault(cat, []).append(pid)
            count += 1
        logger.info("get_eligible_ids_by_category: %d products in %d categories",
                     count, len(by_cat))
        return by_cat

    def get_store_product_ids_map(
        self,
        *,
        min_products: int = 1,
        min_implicit_eligible: int = 0,
    ) -> dict[str, list[str]]:
        """返回合格店铺的 store → [product_id] 映射（结果会缓存）。"""
        return self._store_product_ids_map_cached(min_products, min_implicit_eligible)

    @lru_cache(maxsize=16)
    def _store_product_ids_map_cached(self, min_products, min_implicit_eligible):
        eligible = self._eligible_stores_cached(min_products, min_implicit_eligible)
        if not eligible:
            return {}
        result = {}
        for store in eligible:
            pids = self.get_store_product_ids(
                store=store, min_implicit_eligible=min_implicit_eligible)
            if pids:
                result[store] = pids
        logger.info("get_store_product_ids_map: %d stores, %d total products",
                     len(result), sum(len(v) for v in result.values()))
        return result

    # ── 私有辅助方法 ──

    @staticmethod
    def _build_filter_clauses(
        *,
        min_detail_fields: int = 0,
        min_rating: float | None = None,
        min_rating_number: int | None = None,
        require_price: bool = False,
        require_content: bool = False,
        min_implicit_eligible: int = 0,
        min_reviews: int | None = None,
    ) -> tuple[list[str], list]:
        clauses: list[str] = []
        params: list = []
        if min_detail_fields > 0:
            clauses.append("detail_field_count >= ?")
            params.append(min_detail_fields)
        if min_rating is not None:
            clauses.append("average_rating >= ?")
            params.append(min_rating)
        if min_rating_number is not None:
            clauses.append("rating_number >= ?")
            params.append(min_rating_number)
        if require_price:
            clauses.append("has_price = 1")
        if require_content:
            clauses.append("(has_features = 1 OR has_description = 1)")
        if min_implicit_eligible > 0:
            clauses.append("implicit_eligible_count >= ?")
            params.append(min_implicit_eligible)
        if min_reviews is not None and min_reviews > 0:
            clauses.append("review_count >= ?")
            params.append(min_reviews)
        return clauses, params
