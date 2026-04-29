---
title: surgical-casetime-llm
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>⏱️ Surgical Case Time Prediction — LLM Pipeline</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3b82f6&center=true&vCenter=true&width=700&lines=LangChain+%2B+LangGraph+Orchestration;Gemini+%2B+Groq+Llama+%2B+HuggingFace+Embeddings;Optuna+HPO+%C2%B7+6+Encoding+Strategies+%C2%B7+7+ML+Models" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-4f46e5?style=for-the-badge)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Pipeline-3b82f6?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.1-ffcc00?style=for-the-badge)](https://console.groq.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**⏱️ Surgical Case Time Prediction — LLM Pipeline** — Predicts surgical case duration from pre-operative data using LangChain text embeddings (Gemini API + HuggingFace), Groq Llama-3 LLM feature extraction, and classical ML models with Optuna hyperparameter tuning — all orchestrated automatically by LangGraph.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Pipeline Stages](#-pipeline-stages)
- [Encoding Strategies](#-encoding-strategies)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Outputs & Artifacts](#-outputs--artifacts)
- [Reproducibility](#-reproducibility)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🔗 <b>LangGraph Orchestration</b></td>
    <td>All pipeline stages wired as a LangGraph DAG — each stage auto-skips if already complete</td>
  </tr>
  <tr>
    <td>🤖 <b>Free LLM Integration</b></td>
    <td>Gemini API (free tier) for text embeddings · Groq Llama-3.1-8b (free tier) for structured clinical feature extraction</td>
  </tr>
  <tr>
    <td>🏠 <b>Local HuggingFace Embeddings</b></td>
    <td><code>all-MiniLM-L6-v2</code> runs fully locally — no API key required</td>
  </tr>
  <tr>
    <td>🧬 <b>LLM-Extracted Clinical Features</b></td>
    <td>Groq Llama extracts body region, complexity score, and binary surgical flags from free-text procedure names</td>
  </tr>
  <tr>
    <td>⚙️ <b>Optuna Hyperparameter Tuning</b></td>
    <td>TPE-based search with early stopping across all 7 models and 6 encoding combinations</td>
  </tr>
  <tr>
    <td>📊 <b>Comprehensive Metrics</b></td>
    <td>MAE, RMSE, R², SMAPE, feature importance, and Optuna hyperparameters stored per fold × encoding × model</td>
  </tr>
</table>

---

## 🏗️ Architecture

<div align="center">
<svg width="680" height="275" viewBox="0 0 680 275" xmlns="http://www.w3.org/2000/svg">
  <rect width="680" height="275" rx="12" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
  <text x="340" y="22" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" font-weight="bold" fill="#1e293b">Surgical Case Time LLM Pipeline</text>
  <rect x="20" y="30" width="115" height="55" rx="8" fill="#3b82f6"/>
  <text x="77" y="54" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">casetime</text>
  <text x="77" y="72" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#bfdbfe">.csv</text>
  <rect x="155" y="30" width="170" height="55" rx="8" fill="#4f46e5"/>
  <text x="240" y="54" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Stage 02</text>
  <text x="240" y="72" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#c7d2fe">Gemini / HF Embeddings</text>
  <rect x="355" y="30" width="175" height="55" rx="8" fill="#4f46e5"/>
  <text x="442" y="54" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Stage 02b</text>
  <text x="442" y="72" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#c7d2fe">Groq Llama-3.1 Features</text>
  <rect x="90" y="130" width="500" height="55" rx="8" fill="#0f766e"/>
  <text x="340" y="154" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Stage 03 — Feature Engineering</text>
  <text x="340" y="172" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#99f6e4">5-fold CV · impute · one-hot · PCA</text>
  <rect x="190" y="207" width="300" height="55" rx="8" fill="#7c3aed"/>
  <text x="340" y="231" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="white">Stage 04 — Optuna HPO</text>
  <text x="340" y="249" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#ede9fe">Ridge · Lasso · RF · XGB · LGB · MLP</text>
  <line x1="135" y1="57" x2="147" y2="57" stroke="#64748b" stroke-width="1.5"/>
  <polygon points="147,53 155,57 147,61" fill="#64748b"/>
  <line x1="325" y1="57" x2="347" y2="57" stroke="#64748b" stroke-width="1.5"/>
  <polygon points="347,53 355,57 347,61" fill="#64748b"/>
  <line x1="240" y1="85" x2="240" y2="110" stroke="#64748b" stroke-width="1.5"/>
  <line x1="442" y1="85" x2="442" y2="110" stroke="#64748b" stroke-width="1.5"/>
  <line x1="240" y1="110" x2="442" y2="110" stroke="#64748b" stroke-width="1.5"/>
  <line x1="341" y1="110" x2="341" y2="122" stroke="#64748b" stroke-width="1.5"/>
  <polygon points="337,122 341,130 345,122" fill="#64748b"/>
  <line x1="340" y1="185" x2="340" y2="199" stroke="#64748b" stroke-width="1.5"/>
  <polygon points="336,199 340,207 344,199" fill="#64748b"/>
</svg>
</div>
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- `data/casetime.csv` (source dataset — not included in repo)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/surgical-casetime-llm.git
cd surgical-casetime-llm

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### API Keys

| Service | URL | Used for |
|---------|-----|----------|
| Google Gemini | https://aistudio.google.com | Stage 02 — text embeddings |
| Groq | https://console.groq.com | Stage 02b — LLM feature extraction |
| HuggingFace | *(no key required)* | Stage 02 — local embeddings |

```bash
# Copy the example env file and fill in your keys
cp .env.example .env
```

Or set environment variables directly:

```bash
# Windows
set GEMINI_API_KEY=your_key
set GROQ_API_KEY=your_key

# Mac/Linux
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
```

### Run

```bash
python pipeline.py
```

The LangGraph pipeline runs all stages in order, auto-skipping any already complete.  
To re-run a specific stage, delete its output file (see table below).

---

## 📊 Pipeline Stages

| Stage | Description | Delete to re-run |
|-------|-------------|------------------|
| 01 Data Cleaning | Ingest CSV, validate, persist Clean table | `data/surgical_data.db` |
| 02 LangChain Embeddings | Gemini API + HuggingFace local → embed_cache/ | `data/embed_cache/*.npy` |
| 02b LLM Features | Groq Llama-3.1-8b → structured clinical features | `data/llm_features.json` |
| 03 Feature Engineering | 5-fold CV · impute · one-hot · PCA | `data/fold_encoded.db` |
| 04 Model Training | Optuna HPO · 7 models × 6 encodings | `results/result.db` |

---

## 🔀 Encoding Strategies

| Encoding | Features |
|----------|----------|
| `only_structured` | Tabular features only (baseline) |
| `only_llm` | Tabular + Groq LLM-extracted clinical features |
| `gemini` | Tabular + Gemini embeddings (PCA 768→384) |
| `huggingface` | Tabular + HuggingFace embeddings (384-d) |
| `gemini_llm` | Tabular + Gemini + LLM features |
| `huggingface_llm` | Tabular + HuggingFace + LLM features |

### LLM-Extracted Features (Stage 02b)

For each unique procedure name, Groq Llama-3.1 extracts:

| Feature | Type | Description |
|---------|------|-------------|
| `body_region` | One-hot (10 categories) | abdomen, orthopaedic, neuro, cardio, … |
| `complexity` | Float [0, 1] | 1–5 complexity score, normalised |
| `is_bilateral` | Binary | Bilateral procedure flag |
| `is_laparoscopic` | Binary | Minimally invasive flag |
| `is_robotic` | Binary | Robotic-assisted flag |
| `is_emergency` | Binary | Emergency/urgent procedure flag |
| `n_procedures` | Float | Procedure count, normalised |

---

## 🧠 ML Models

```python
# All models tuned with Optuna TPE (N_TRIALS = 20 per combo)
models = {
    "ridge":          "Ridge Regression (Optuna α)",
    "lasso":          "Lasso Regression (Optuna α)",
    "random_forest":  "RandomForestRegressor (Optuna n_est, depth, features)",
    "xgboost":        "XGBRegressor with early stopping (Optuna)",
    "lightgbm":       "LGBMRegressor (Optuna)",
    "mlp":            "TensorFlow/Keras MLP, AdamW, BatchNorm, depth 1–3 (Optuna)",
}
```

---

## 📁 Project Structure

```
surgical-casetime-llm/
│
├── 📄 pipeline.py              # Main LangGraph pipeline (Stages 01–04)
├── 📄 requirements.txt         # Python dependencies
├── 📄 .env.example             # API key template
│
├── 📂 data/
│   ├── 📂 embed_cache/         # Gemini and HuggingFace embedding cache (.npy)
│   └── 📄 llm_features.json    # Groq-extracted clinical features per procedure
│
├── 📂 results/                 # result.db, per-model logs and PDF summaries
│
├── 📂 overleaf/                # LaTeX manuscript
│   └── 📄 *.tex                # Paper sections
│
├── 📂 flowchart/               # Pipeline flowchart assets
└── 📄 sync.ps1                 # Git sync utility
```

---

## 📦 Outputs & Artifacts

Results are saved to `results/result.db` (SQLite) with tables:

| Table | Contents |
|-------|----------|
| `metrics` | MAE, RMSE, R², SMAPE per fold × encoding × model |
| `predictions` | Actual vs predicted duration per surgical case |
| `feature_importance` | Coefficients / importances / gradient saliency |
| `hyperparameter` | Best Optuna parameters per combination |

Per-model `.log` and `.pdf` summary files are also saved to `results/`.

---

## 🔁 Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`) used throughout all stages.
- All preprocessing (imputation, encoding, PCA) is fit on train folds only — no leakage.
- Gemini and HuggingFace embeddings are cached after the first run.
- LLM feature extraction (Stage 02b) is cached in `llm_features.json` — deterministic per procedure name.
- Stages 01–03 auto-skip when outputs are present.

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional medical advice of any kind. All datasets used are subject to institutional data-use agreements — no patient-identifiable information is included in this repository. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/surgical-casetime-llm?style=social)](https://github.com/mnoorchenar/surgical-casetime-llm)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/surgical-casetime-llm?style=social)](https://github.com/mnoorchenar/surgical-casetime-llm/fork)

<sub>This project is developed purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>
