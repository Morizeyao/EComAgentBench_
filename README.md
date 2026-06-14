# EComAgentBench

**Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent**

EComAgentBench evaluates whether an LLM shopping agent can *actively gather* a user's needs that
are deliberately scattered across three sources, and recommend the right product. It ships a
fully automated, three-stage pipeline:

1. **Generation** — automatically construct verifiable benchmark samples from a real product
   catalog (Amazon Reviews 2023).
2. **Prediction** — let an agent interact with an API sandbox (search, filter, reviews, user
   profile, clarification, …) and recommend exactly one product.
3. **Evaluation** — score the recommendation with rubric-based LLM-as-Judge plus exact-match.

This repository contains the **662 validated benchmark samples**, the product-database build
scripts, and the full generation / prediction / evaluation code used in the paper.

---

## Repository structure

```text
.
├── README.md
├── DATA.md                       Data sources, license, synthetic-persona statement
├── LICENSE                       MIT (code only)
├── pyproject.toml
├── .env.example                  Per-vendor API key template
├── configs/
│   └── settings.yaml             Main config for all three stages
├── data/
│   ├── benchmark/
│   │   └── benchmark.jsonl        662 validated samples
│   └── products/                  product.db is built here (not shipped)
├── scripts/
│   ├── download_amazon_reviews.py Download raw Amazon Reviews 2023 from the HF Hub
│   ├── build_product_index.py     Build SQLite + FTS5 product/review index
│   ├── generate.sh / predict.sh / evaluate.sh
└── src/
    ├── database/                  SQLite + FTS5 product DB wrapper
    ├── generation/                Benchmark construction pipeline
    ├── prediction/                Shopping agent + batch runner
    ├── evaluation/                Rubric judge + metrics
    ├── llm/                       Unified LLM facade (OpenAI / Anthropic / Gemini / OpenRouter)
    ├── tools/                     The 10 agent tools
    ├── prompts/                   System prompts
    └── intents/                   Intent catalog
```

---

## Installation

Requires Python `>=3.11,<3.12`.

With [uv](https://github.com/astral-sh/uv) (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

---

## API keys

This open-source version uses each vendor's **official API** (no custom gateway). Copy the
template and fill in the keys for the providers you intend to use:

```bash
cp .env.example .env
```

| Variable | Provider | Used by |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI | prediction (`openai_api`) |
| `ANTHROPIC_API_KEY` | Anthropic | prediction (`claude`) |
| `GEMINI_API_KEY` | Google Gemini | generation + evaluation judge + prediction (`gemini`) |
| `OPENROUTER_API_KEY` | OpenRouter | prediction (`openrouter`: Kimi / MiniMax / Qwen / …) |

> Keys are resolved per-vendor in `src/llm/utils.py`. If you need to route a provider through a
> custom gateway/proxy, set `base_url` on the corresponding profile in `configs/settings.yaml`.

---

## Data preparation

`benchmark.jsonl` ships with this repo. The product database `product.db` is large (~25 GB) and
hosted separately — you have **two options** to obtain it.

### Option A — Download the prebuilt database (recommended, ready to use)
The dataset authors have authorized release, so the exact `product.db` used in the paper is
published at the Hugging Face dataset
[`ecomagentbench/EcomAgentBenchProductDB`](https://huggingface.co/datasets/ecomagentbench/EcomAgentBenchProductDB)
(`product.db` at the repo root):
```bash
uv run python scripts/download_db.py
# or override the repo explicitly:
uv run python scripts/download_db.py --repo ecomagentbench/EcomAgentBenchProductDB
```
This downloads `product.db` into `data/products/`. After this, **prediction and evaluation run out
of the box** (just set the API key for the provider you use, see below).

### Option B — Rebuild from Amazon Reviews 2023 (optional, transparent)
The paper builds `product.db` from the **full** version of four categories — `All_Beauty`,
`Electronics`, `Cell_Phones_and_Accessories`, `Office_Products` (~3.72M products / ~21M reviews),
which covers all benchmark target products.

> **Disk & time**: the four full categories are tens of GB to download and slow to index.
> To just try the pipeline, start with one small category (`--categories All_Beauty`).

```bash
# 1) download raw category files from the Hugging Face Hub
uv run python scripts/download_amazon_reviews.py --paper --output-dir data/amazon_raw
#    quick try: --categories All_Beauty   |   everything: --all

# 2) build the SQLite + FTS5 index (products first, then reviews: 20/product, seed 42)
uv run python scripts/build_product_index.py \
  --input data/amazon_raw/products --output data/products/product.db
uv run python scripts/build_product_index.py \
  --reviews-input data/amazon_raw/reviews --output data/products/product.db \
  --max-reviews-per-product 20 --seed 42
```
If `HF_TOKEN` is set in `.env`, it is used automatically to raise the download rate limit.

> **Reproducibility of a rebuilt DB**: the product set is fully determined by the downloaded
> categories, and review sampling uses a stable per-product seed (independent of `PYTHONHASHSEED`),
> so a rebuilt DB is deterministic across machines. It is **not** byte-identical to the original
> paper DB — the specific 20-reviews-per-product subset differs (the original used a non-fixed hash
> seed). **Products / queries / personas / clarifications are reproducible; only review-derived
> content (review evidence, `review_opinion` rubrics) may differ.** This does not affect the
> released benchmark, whose evaluation depends on `benchmark.jsonl` + target coverage, not on the
> exact review subset.

---

## Pipeline

### 1. Generation (optional — `benchmark.jsonl` is already provided)
Construct new benchmark samples from the product DB:
```bash
CONFIG=configs/settings.yaml SAMPLES=10 THREADS=16 bash scripts/generate.sh
# equivalently:
uv run python -m src.generation --config configs/settings.yaml --samples 10
```
Overridable env vars: `OUTPUT`, `SAMPLES` (per-intent count), `INTENTS`, `THREADS`.
Output is written to `data/benchmark/benchmark.jsonl`.

> Note: re-running generation reproduces product / query / persona / clarification content, but
> `review_opinion` rubrics and review evidence depend on the database's per-product review subset,
> which may differ from the original build (see the rebuild reproducibility note above). Everything
> else is stable.

### 2. Prediction
Run an agent over the benchmark:
```bash
CONFIG=configs/settings.yaml MODE=claude bash scripts/predict.sh
# equivalently:
uv run python -m src.prediction.run_predict --config configs/settings.yaml --mode claude
```
Supported `MODE`: `openai_api`, `openrouter`, `gemini`, `claude`.
Only `sample_status == "ok"` samples are run. Each agent gets up to `max_steps` (default 100)
tool calls and must end with `recommend_product`. Output:
`data/benchmark/predictions_{model}.jsonl` (full `trajectory` + `reasoning_trace` + tool-call stats).

### 3. Evaluation
Score predictions with rubric-based LLM-as-Judge (Gemini):
```bash
CONFIG=configs/settings.yaml bash scripts/evaluate.sh
# or score a specific predictions file:
bash scripts/evaluate.sh --predictions data/benchmark/predictions_claude-opus-4-6.jsonl
# equivalently:
uv run python -m src.evaluation.run_eval --config configs/settings.yaml
```
Outputs (next to the predictions file):
`predictions_{model}_evaluation_results.jsonl`, `_evaluation_summary.json`, `_evaluation_summary.md`.

---

## Configuration

Everything lives in `configs/settings.yaml`. The blocks you will touch most:

- `data.db_path` / `data.benchmark_dir` — data locations.
- `generation.*` — `seed`, `min_features`/`max_features`, `partition.*` ratios,
  `clarification.max_turns`, per-`intents` counts, `llm.model`.
- `prediction.mode` and `prediction.modes.*` — pick the provider/model; `max_steps`.
- `evaluation.llm.model` — the judge model.

`llm_profiles` map a logical name (`openai` / `gemini` / `claude` / `openrouter`) to a provider.

---

---

## Evaluation metrics

The evaluation summary reports, among others:

- `overall_accuracy` — `exact_match OR all_rubrics_satisfied`.
- `exact_match_rate` and non-exact-but-correct rate.
- `overall_rubric_satisfaction`, broken down **by source** (`query` / `persona` / `clarification`)
  and **by rubric type** (`attribute_match`, `numeric_range`, `entity_match`,
  `negative_attribute`, `budget_match`, `review_opinion`).
- `by_intent` — accuracy per intent category.
- Behavioral stats: finish rate, tool-call counts, persona-utilization rate, clarification usage.

---

## Notes on official APIs

This release targets the **official** OpenAI / Anthropic / Google / OpenRouter APIs (keys read per
vendor in `.env`; no custom gateway). Provider-specific handling:

- **Anthropic (Claude)** — with extended thinking enabled (default `thinking_budget` in
  `configs/settings.yaml`), the official API requires thinking blocks to be echoed back before
  `tool_use` in multi-turn tool calling. The Claude provider does this automatically.
- **OpenAI (gpt-5)** — uses the Responses API; reasoning items are carried across turns.
- **Gemini (gemini-3.x)** — function-call `id`s are echoed back in function responses (required by
  Gemini 3). In the paper Gemini is used only as judge / generator (no tool calling), so the Gemini
  *agent* path is newly enabled here.
- **OpenRouter** — OpenAI-compatible; used in the paper for Kimi / MiniMax / Qwen.

**Validation status.** The Gemini / Claude / GPT results in the paper were produced through an
internal API gateway; due to access constraints we could not validate these three providers
against their public official APIs. The provider code here targets the official APIs as a
best-effort adaptation and **may require minor changes to run end-to-end** against a given
official API (e.g. evolving model IDs, request fields, or SDK versions). The **OpenRouter** path
has been validated end-to-end (prediction + evaluation). For OpenAI / Anthropic / Gemini, run a
quick **smoke test (1–2 samples)** with your own key before a full run.

## Benchmark schema

Each line of `data/benchmark/benchmark.jsonl` is a sample. Key fields used at evaluation time:

| Field | Description |
| --- | --- |
| `id` | Sample id |
| `intent_category` | One of 8 intents |
| `user_query` | Visible initial request |
| `user_persona` | Synthetic profile; `product_requirements` holds the scored persona constraints |
| `clarification_script` | Deterministic slots: `trigger_keywords` → `user_response`, with `linked_rubric_ids` |
| `target_product` | The reference product (`product_id`, `title`, `details`, …) |
| `rubrics` | List of `{id, type, field, expected_value, info_source}` checks |
| `info_source` (per rubric) | `query` / `persona` / `clarification` |
| `sample_status` | `"ok"` for all shipped samples |

Other fields (`extracted_features`, `feature_partition`, `validation_*`, `judge_*`, …) are
generation artifacts retained for transparency. See [DATA.md](DATA.md).

---

## License & data

- **Code**: MIT — see [LICENSE](LICENSE).
- **Data**: the Amazon Reviews 2023 corpus is downloaded separately and is for **research use
  only**; personas are synthetic with no PII. See [DATA.md](DATA.md).

---

## Citation

```bibtex
@inproceedings{ecomagentbench2026,
  title     = {EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026 (Industry Track)},
  year      = {2026}
}
```

Please also cite the underlying dataset (Hou et al., 2024); see [DATA.md](DATA.md).
