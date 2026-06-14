"""商品采样器：均匀类别采样。"""

import logging
import random
from collections import defaultdict

from src.generation.config import (
    get_min_implicit_count,
    requires_content_evidence,
    requires_price,
    UNDERUSED_FEATURE_POOL,
)

logger = logging.getLogger(__name__)


class ProductSampler:
    """均匀类别采样：过滤小类别后等额分配。"""

    def __init__(self, db, seed: int = 42, min_category_size: int = 30):
        self.db = db
        self.rng = random.Random(seed)
        self.min_category_size = min_category_size
        self._feature_usage: dict[str, int] = defaultdict(int)

    def sample_batch(self, intent: dict, batch_size: int = 200) -> list[dict]:
        intent_id = intent["id"]
        strategy = intent.get("sampling_strategy", {})
        return self._sample_uniform_batch(intent_id, strategy, batch_size)

    def _sample_uniform_batch(self, intent_id: str, strategy: dict,
                              batch_size: int) -> list[dict]:
        ids_by_cat = self.db.get_eligible_ids_by_category(
            min_detail_fields=strategy.get("min_detail_fields", 6),
            min_rating=strategy.get("min_rating"),
            min_rating_number=strategy.get("min_rating_number"),
            require_price=requires_price(intent_id),
            require_content=requires_content_evidence(intent_id),
            min_implicit_eligible=get_min_implicit_count(intent_id),
            min_reviews=strategy.get("min_reviews"),
        )
        if not ids_by_cat:
            return []

        # 过滤小类别，然后均匀分配
        cats = [
            (cat, ids) for cat, ids in ids_by_cat.items()
            if len(ids) >= self.min_category_size
        ]
        if not cats:
            cats = list(ids_by_cat.items())

        n_cats = len(cats)
        per_cat = max(1, batch_size // n_cats)

        selected_ids = []
        for _cat, ids in cats:
            k = min(per_cat, len(ids))
            selected_ids.extend(self.rng.sample(ids, k))

        products = self.db.get_products(selected_ids)

        require_fields = strategy.get("require_details_fields", [])
        if require_fields:
            products = [
                p for p in products
                if all(f in p.get("details", {}) for f in require_fields)
            ]
        return products

    def update_feature_usage(self, features: list[str]):
        for f in features:
            self._feature_usage[f] += 1

    def get_underused_features(self, top_n: int = 5) -> list[str]:
        sorted_feats = sorted(
            UNDERUSED_FEATURE_POOL,
            key=lambda f: self._feature_usage.get(f, 0),
        )
        return sorted_feats[:top_n]
