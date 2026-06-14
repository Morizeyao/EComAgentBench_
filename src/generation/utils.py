"""文本与数值工具函数。"""

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
ENTITY_STOPWORDS = {"the", "a", "an", "for", "with", "and", "or", "of"}


def normalize_text(text: object) -> str:
    """全部小写并提取字母/数字 token，返回空格分隔的字符串。"""
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def normalize_token(token: str) -> str:
    """简单英文复数还原（ies → y, s → 去掉）。"""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def keyword_tokens(text: object) -> list[str]:
    """提取去停用词、做简单词干还原的关键词 token 列表。"""
    return [
        normalize_token(t)
        for t in TOKEN_RE.findall(str(text).lower())
        if t not in ENTITY_STOPWORDS
    ]


# ── 数值约束工具 ──

MIN_OP_TERMS = ("more than", "over", "above")
MIN_EQ_TERMS = ("at least", "or better", "minimum", "no less than")
MAX_OP_TERMS = ("less than", "below", "under")
MAX_EQ_TERMS = ("at most", "no more than")


def normalize_numeric_expected(description: str, expected: object) -> object:
    """根据 description 中的关键词把标量 expected 转为结构化 min/max/op 对象。"""
    if not isinstance(expected, (int, float)):
        return expected

    lower_desc = description.lower()
    if any(t in lower_desc for t in MIN_OP_TERMS):
        return {"value": expected, "op": ">"}
    if any(t in lower_desc for t in MIN_EQ_TERMS):
        return {"min": expected}
    if any(t in lower_desc for t in MAX_OP_TERMS):
        return {"value": expected, "op": "<"}
    if any(t in lower_desc for t in MAX_EQ_TERMS):
        return {"max": expected}
    return expected
