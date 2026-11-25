# ERCR RAG

Official implementation of **"ERCR RAG: Robust Retrieval Augmented Generation via Evidence Refinement and Conflict Resolution"** (2026).

ERCR RAG is a **training-free** defense framework that hardens RAG pipelines against corpus-poisoning and prompt-injection attacks (PIA, PoisonedRAG, AD). It combines fine-grained evidence refinement with dual-source knowledge conflict resolution:

1. **Sentence-Level Screening (SLS)** – dual-threshold similarity band (`0.42 < sim < 0.81`) over context-aware sentence embeddings (`alpha = 0.9`) removes both irrelevant and over-similar adversarial sentences.
2. **Semantic-Facilitated Purification (SFP)** – bait-guided k-means clustering (`k = 3`) plus intra-cluster lexical-diversity checks (cosine 0.88 / ROUGE-L 0.25) excises adversarial clusters.
3. **Dual-Source Reasoning Pipeline (DSRP)** – three low-temperature LLM calls reconcile purified external evidence with the model's parametric knowledge before answering.
4. **ERCR-PLUS** (optional plug-in, high-threat settings) – target-answer classification, temporal-logic perception, and entity-relation-count reordering.

## Requirements

- Python 3.10+, CUDA-capable GPU (tested on NVIDIA A800)
- LLM: Llama-3.1-8B-Instruct (default), Mistral-NeMo-12B, etc.
- Retriever: Contriever (default), Contriever-MS, or ANCE
- Sentence embeddings: SimCSE (`princeton-nlp/sup-simcse-bert-base-uncased`, default)

```bash
pip install torch transformers sentence-transformers beir nltk spacy scikit-learn \
            rouge-score loguru tqdm umap-learn pandas matplotlib seaborn
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt punkt_tab
```

## Model Paths

All checkpoints default to their Hugging Face repo ids and can be overridden via environment variables (useful for local mirrors):

| Variable | Default |
|---|---|
| `ERCR_LLM_PATH` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `ERCR_EMBEDDING_MODEL` | `princeton-nlp/sup-simcse-bert-base-uncased` |
| `ERCR_CONTRIEVER_PATH` | `facebook/contriever` |
| `ERCR_CONTRIEVER_MSMARCO_PATH` | `facebook/contriever-msmarco` |
| `ERCR_ANCE_PATH` | `sentence-transformers/msmarco-roberta-base-ance-firstp` |
| `ERCR_NLTK_DATA` | *(optional local NLTK data dir)* |

> Note: Llama models require accepting the Meta license on Hugging Face and setting `HF_TOKEN`.

## Data Preparation

The loader expects the following layout (paths relative to the repository root):

```
../data_cache/                       # BEIR corpus/queries/qrels pickles (auto-cached)
../results/beir_results/             # pre-computed BEIR retrieval runs, e.g.
                                     #   {dataset}-contriever.json
../results/adv_targeted_results/     # evaluation targets and adversarial texts:
                                     #   {dataset}.json                  (target queries w/ correct & incorrect answers)
                                     #   {dataset}_DeepSeek(V3).json     (pre-generated adversarial texts)
```

Supported `{dataset}` values: `hotpotqa`, `nq`, `msmarco`.

## Usage

Main entry point: `ERCR_RAG.py`. Key flags:

- `--attack_method`: `none` (clean), `pia` (PIA), `hotflip` (PoisonedRAG), `LM_targeted` (AD; requires pre-generated adversarial texts)
- `--removal_method`: `evidence_R` (SLS + SFP), `PLUS` (SLS + SFP + PLUS), `kmeans_ngram` (TrustRAG stage 1&2), `none`
- `--defend_method`: `DSRP` (ERCR conflict resolution), `Trust` (TrustRAG), `none` (vanilla single-call generation)
- `--eval_model_code`: `contriever` (default), `contriever-msmarco`, `ance`
- `--top_k 5 --adv_per_query 1` reproduces the 20% poisoning rate of the paper; `--M 10 --repeat_times 10` reproduces the 100-query evaluation.

### Examples

```bash
# Vanilla RAG (clean)
python ERCR_RAG.py --eval_dataset hotpotqa --attack_method none \
    --removal_method none --defend_method none

# ERCR RAG under PoisonedRAG (20% poisoning)
python ERCR_RAG.py --eval_dataset hotpotqa --attack_method hotflip \
    --removal_method evidence_R --defend_method DSRP

# ERCR-PLUS RAG under PIA
python ERCR_RAG.py --eval_dataset hotpotqa --attack_method pia \
    --removal_method PLUS --defend_method DSRP

# TrustRAG baseline
python ERCR_RAG.py --eval_dataset hotpotqa --attack_method hotflip \
    --removal_method kmeans_ngram --defend_method Trust
```

Baseline reproductions live in `baselines/` (TrustRAG, SeConRAG, EcoSafeRAG, GMTP).

## Outputs & Metrics

Per run, artifacts are written to `outputs/{log_name}/`: `top_ks`, `final_answers`, `agents_drafts` (DSRP intermediates). Logs report:

- **ACC** – share of answers containing the ground-truth answer (substring protocol)
- **ASR** – share of answers containing the adversarial target while missing the ground truth
- **AIR / RR** – attack injection rate and retention rate of the first-stage filter

Default configuration follows the paper: seed 12, temperature 0.01, max 1024 new tokens, deterministic decoding (`do_sample=False` for bait drafting).

## Repository Structure

```
ERCR_RAG.py         # main experiment pipeline (attack -> refinement -> DSRP -> eval)
defend_module.py    # SLS, SFP, bait filtering, DSRP prompts, TrustRAG/ASTUTE helpers
defend_plugin.py    # PLUS: answer classification, temporal logic, relation reordering
src/attack.py       # PIA / PoisonedRAG / AD attack implementations
src/prompts.py      # QA prompt templates
src/utils.py        # BEIR data loading, retriever loading, caching
utils_Llama.py      # local HF LLM wrapper (APILLM)
baselines/          # baseline reproductions
```

## License

This codebase is released for research purposes. The underlying models (Llama, Contriever, ANCE, SimCSE), datasets (HotpotQA, NQ, MS-MARCO), and libraries are subject to their respective licenses.
