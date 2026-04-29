# Clinical Surgery Case Time Prediction
### LangChain + LangGraph Pipeline

Predicts surgical case duration from pre-operative data using LangChain text embeddings, Groq LLM feature extraction, and classical ML models with Optuna hyperparameter tuning.

---

## Architecture

```
Stage 01  Data Cleaning        casetime.csv → surgical_data.db
Stage 02  LangChain Embeddings Gemini API (free) + HuggingFace (local) → embed_cache/
Stage 02b LLM Features         Groq Llama-3.1-8b (free) → llm_features.json
Stage 03  Feature Engineering  5-fold CV · impute · one-hot · PCA → fold_encoded.db
Stage 04  Model Training       Optuna HPO · Ridge/Lasso/RF/XGB/LGB/MLP → result.db
```

LangGraph orchestrates all stages automatically — each stage skips itself if already complete.

## Encodings compared in Stage 04

| Encoding | Features |
|---|---|
| `only_structured` | Tabular features only (baseline) |
| `only_llm` | Tabular + LLM-extracted clinical features |
| `gemini` | Tabular + Gemini embeddings (PCA 768→384) |
| `huggingface` | Tabular + HuggingFace embeddings (384-d) |
| `gemini_llm` | Tabular + Gemini + LLM features |
| `huggingface_llm` | Tabular + HuggingFace + LLM features |

## LLM-extracted features (Stage 02b)

For each procedure name, Groq Llama extracts:
- `body_region` — one-hot over 10 categories (abdomen, orthopedic, neuro, …)
- `complexity` — 1–5 scale, normalised
- `is_bilateral`, `is_laparoscopic`, `is_robotic`, `is_emergency` — binary flags
- `n_procedures` — count, normalised

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get free API keys

| Service | URL | Used for |
|---|---|---|
| Google Gemini | https://aistudio.google.com | Stage 02 embeddings |
| Groq | https://console.groq.com | Stage 02b LLM features |
| HuggingFace | *(no key)* | Stage 02 embeddings (local) |

### 3. Set API keys
```bash
# Copy the example and fill in your keys
cp .env.example .env

# Windows
set GEMINI_API_KEY=your_key
set GROQ_API_KEY=your_key

# Mac/Linux
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
```

Or paste them directly into the `CONFIG` block at the top of `pipeline.py`.

### 4. Run
```bash
python pipeline.py
```

The pipeline runs all stages in order, skipping any that are already complete.
To re-run a specific stage, delete its output file:

| Stage | Delete to re-run |
|---|---|
| Stage 01 | `data/surgical_data.db` |
| Stage 02 | `data/embed_cache/*.npy` |
| Stage 02b | `data/llm_features.json` |
| Stage 03 | `data/fold_encoded.db` |
| Stage 04 | `results/result.db` |

---

## Configuration

All tunable settings are in the `CONFIG` block at the top of `pipeline.py`:

- `MODELS_TO_RUN` — subset of models to train (default: all 7)
- `N_SPLITS` — cross-validation folds (default: 5)
- `N_TRIALS` — Optuna trials per model (default: 20)
- `GEMINI_MODEL` — embedding model (default: `text-embedding-004`)
- `HF_MODEL` — local embedding model (default: `all-MiniLM-L6-v2`)
- `GROQ_MODEL` — LLM model (default: `llama-3.1-8b-instant`)

## Output

Results are saved to `results/result.db` (SQLite) with tables:
- `metrics` — MAE, RMSE, R², SMAPE per fold × encoding × model
- `predictions` — actual vs predicted per case
- `feature_importance` — coefficients / importances / gradient saliency
- `hyperparameter` — best Optuna params per combo

Per-model `.log` and `.pdf` summary files are saved to `results/`.
