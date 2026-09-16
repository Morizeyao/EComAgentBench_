# EComAgentBench

**Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent**

**Accepted to EMNLP 2026 Industry Track.** Camera-ready version submitted.

**Zeyao Du, Tong Li, Haibo Zhang** · Shopee

[Paper (arXiv)](https://arxiv.org/abs/2606.17698v3)
[Product database](https://huggingface.co/datasets/ecomagentbench/EcomAgentBenchProductDB) ·
[Citation](#citation)

EComAgentBench evaluates whether an LLM shopping agent can *actively gather* a user's needs that
are deliberately scattered across three sources, and recommend the right product. It ships a
fully automated, three-stage pipeline:

1. **Generation** — automatically construct verifiable benchmark samples from a real product
   catalog (Amazon Reviews 2023).
2. **Prediction** — let an agent interact with an API sandbox (search, filter, reviews, user
   profile, clarification, …) and recommend exactly one product.
3. **Evaluation** — score the recommendation with rubric-based LLM-as-Judge plus exact-match.

This repository contains the **662 validated benchmark samples**, the product-database build
scripts, and the generation / prediction / evaluation pipeline with public API adapters.

## Benchmark at a glance

| Component | Released benchmark |
| --- | --- |
| Tasks | 662 validated tasks across 8 shopping intents |
| Requirements | 6,645 typed rubrics across 6 rubric types |
| Information sources | Visible query, tool-gated persona, scripted clarification |
| Environment | 10 tools for product search, inspection, reviews, user interaction, and recommendation |
| Episode budget | 100 agent iterations, at most one tool call per iteration, and up to 10 clarification turns |
| Success criterion | Exact target-product match **or** satisfaction of every rubric |

The released tasks are in English, cover a single marketplace with electronics-adjacent
categories, and end in one product recommendation. Personas are synthetic and clarification
responses are deterministic. The results characterize this setting rather than all e-commerce
domains.

## Paper results

The following are the frozen results reported in the
[camera-ready paper](https://arxiv.org/abs/2606.17698v3), evaluated on the same 662 tasks.

| Model | Accuracy (%) | Rubric satisfaction (%) | Finish (%) | Avg. tool calls |
| --- | ---: | ---: | ---: | ---: |
| Claude Opus 4.6 | 57.1 | 76.6 | 84.6 | 33.9 |
| GPT-5.4 | 47.0 | 81.0 | 99.4 | 25.1 |
| Kimi K2.6 | 46.4 | 73.4 | 88.7 | 38.2 |
| GPT-5 | 39.6 | 76.1 | 97.7 | 24.4 |
| MiniMax M2.7 | 36.6 | 73.5 | 95.6 | 31.3 |
| GPT-5 mini | 31.6 | 72.7 | 100.0 | 15.5 |
| Qwen3-30B-A3B | 19.5 | 59.0 | 100.0 | 18.3 |

Accuracy measures task-level success under the criterion above; rubric satisfaction measures
the fraction of individual requirements satisfied. Finish is the harness's `finished` flag
and does not necessarily imply a valid recommendation. These are the original paper results,
not fresh runs against current public API endpoints; see [API validation status](#notes-on-official-apis).

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
This downloads `product.db` into `data/products/`. Configure the API keys for your prediction
provider and the Gemini evaluation judge before running the pipeline. See
[API validation status](#notes-on-official-apis) for the tested and unvalidated provider paths.

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
> seed). The released `benchmark.jsonl` remains fixed, but a different review subset can change
> the evidence retrieved by agents and the review evidence available to the evaluation judge.
> Use the prebuilt database together with the released benchmark to match the paper's task
> environment. Re-running LLM-based generation does not guarantee identical samples.

---

## Pipeline

### 1. Generation (optional — `benchmark.jsonl` is already provided)
Construct new benchmark samples from the product DB:
```bash
CONFIG=configs/settings.yaml OUTPUT=data/generated/benchmark.jsonl \
  SAMPLES=10 THREADS=16 bash scripts/generate.sh
# equivalently:
uv run python -m src.generation --config configs/settings.yaml --samples 10 \
  --output data/generated/benchmark.jsonl
```
Overridable env vars: `OUTPUT`, `SAMPLES` (per-intent count), `INTENTS`, `THREADS`.
The examples write new samples to `data/generated/benchmark.jsonl`. Omitting `OUTPUT` or
`--output` uses `data/benchmark/benchmark.jsonl` and overwrites the released benchmark.

> Generation constructs a new dataset using LLM calls; it is not required to evaluate the
> released 662 tasks. To evaluate a newly generated dataset, point `data.benchmark_dir` in a
> separate configuration file to its directory.

### 2. Prediction
Run an agent over the benchmark:
```bash
CONFIG=configs/settings.yaml MODE=claude bash scripts/predict.sh
# equivalently:
uv run python -m src.prediction.run_predict --config configs/settings.yaml --mode claude
```
Supported `MODE`: `openai_api`, `openrouter`, `gemini`, `claude`.
Only `sample_status == "ok"` samples are run. Each agent gets up to `max_steps` (default 100)
agent iterations, with at most one executed tool call per iteration; turns without a tool call
also consume an iteration. A valid completed task requires a `recommend_product` call. Output:
`data/benchmark/predictions_{model}.jsonl` (full `trajectory` + `reasoning_trace` + tool-call stats).

### 3. Evaluation
Score predictions with rubric-based LLM-as-Judge (Gemini):
```bash
CONFIG=configs/settings.yaml bash scripts/evaluate.sh \
  --predictions data/benchmark/predictions_claude-opus-4-6.jsonl
# equivalently:
uv run python -m src.evaluation.run_eval --config configs/settings.yaml \
  --predictions data/benchmark/predictions_claude-opus-4-6.jsonl
```
Replace the predictions path with the output of your run. A `MODE` override used for prediction
does not change the mode saved in the configuration; passing `--predictions` explicitly ensures
that evaluation reads the intended file.

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

Please cite the [arXiv paper](https://arxiv.org/abs/2606.17698).

```bibtex
@misc{du2026ecomagentbench,
  title         = {EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent},
  author        = {Zeyao Du and Tong Li and Haibo Zhang},
  year          = {2026},
  eprint        = {2606.17698},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.17698}
}
```

Please also cite the underlying dataset (Hou et al., 2024); see [DATA.md](DATA.md).
