"""
make_figures.py
Generates all manuscript figures from results/result.db.
Run: python make_figures.py
Outputs: results/figures/fig_*.pdf  (+ .png thumbnails)
"""

import os, ast, sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

DB_PATH  = "results/result.db"
OUT_DIR  = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
ENC_COLORS = {
    "only_structured":  "#7F8C8D",
    "only_llm":         "#E67E22",
    "huggingface":      "#2980B9",
    "huggingface_llm":  "#27AE60",
}
ENC_LABELS = {
    "only_structured":  "Structured only (38-d)",
    "only_llm":         "Structured + LLM (54-d)",
    "huggingface":      "Structured + HuggingFace (422-d)",
    "huggingface_llm":  "Structured + HF + LLM (438-d)",
}
MODEL_ORDER = ["ridge", "lasso", "elasticnet", "randomforest", "xgboost", "lightgbm", "mlp"]
MODEL_LABELS = {
    "ridge":        "Ridge",
    "lasso":        "Lasso",
    "elasticnet":   "ElasticNet",
    "randomforest": "Random Forest",
    "xgboost":      "XGBoost",
    "lightgbm":     "LightGBM",
    "mlp":          "MLP",
}
ENC_ORDER = ["only_structured", "only_llm", "huggingface", "huggingface_llm"]

def load_metrics():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM metrics", con)
    con.close()
    return df

def load_predictions():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", con)
    con.close()
    return df

def load_feature_importance():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM feature_importance", con)
    con.close()
    return df

def savefig(fig, name, dpi=200):
    pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    png = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=dpi)
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {pdf}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Main results: MAE + R² heatmap (encoding × model)
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap(df):
    agg = df.groupby(["encoding", "model"])[["mae", "r2"]].mean().reset_index()

    mae_pivot = agg.pivot(index="encoding", columns="model", values="mae")
    r2_pivot  = agg.pivot(index="encoding", columns="model", values="r2")
    mae_pivot = mae_pivot.loc[ENC_ORDER, MODEL_ORDER]
    r2_pivot  = r2_pivot.loc[ENC_ORDER, MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("5-Fold Cross-Validation Performance\n(Encoding × Model)", fontsize=13, fontweight="bold")

    for ax, pivot, title, cmap, fmt, best_fn in [
        (axes[0], mae_pivot, "Mean Absolute Error (min) — lower is better",
         "YlOrRd_r", ".1f", np.argmin),
        (axes[1], r2_pivot,  "R² Score — higher is better",
         "YlGn",    ".3f", np.argmax),
    ]:
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(ENC_ORDER)))
        ax.set_yticklabels([ENC_LABELS[e] for e in ENC_ORDER], fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < 0.9 * pivot.values.max() else "white")

        # highlight best cell
        best_flat = best_fn(pivot.values.flatten())
        bi, bj = divmod(best_flat, pivot.shape[1])
        ax.add_patch(mpatches.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                         fill=False, edgecolor="#C0392B", linewidth=2.5))

    fig.tight_layout()
    savefig(fig, "fig1_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Grouped bar chart: MAE by model, grouped by encoding
# ─────────────────────────────────────────────────────────────────────────────
def fig_grouped_bar(df):
    agg = df.groupby(["encoding", "model"])["mae"].agg(["mean", "std"]).reset_index()

    x     = np.arange(len(MODEL_ORDER))
    width = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(13, 5))
    for enc, off in zip(ENC_ORDER, offsets):
        sub = agg[agg["encoding"] == enc].set_index("model")
        means = [sub.loc[m, "mean"] if m in sub.index else np.nan for m in MODEL_ORDER]
        stds  = [sub.loc[m, "std"]  if m in sub.index else 0      for m in MODEL_ORDER]
        bars = ax.bar(x + off * width, means, width, yerr=stds,
                      label=ENC_LABELS[enc], color=ENC_COLORS[enc],
                      alpha=0.85, capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
    ax.set_ylabel("Mean Absolute Error (minutes)", fontsize=11)
    ax.set_title("MAE by Model and Encoding Strategy\n(mean ± std across 5 folds)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.axhline(26.58, color="#C0392B", linestyle="--", linewidth=1.2, label="Best: 26.58 min")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig2_grouped_bar")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Boxplots: fold-level variability per encoding (best models)
# ─────────────────────────────────────────────────────────────────────────────
def fig_fold_boxplots(df):
    top_models = ["xgboost", "lightgbm", "mlp", "randomforest"]
    sub = df[df["model"].isin(top_models)].copy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    fig.suptitle("5-Fold MAE Distribution by Encoding\n(top 4 models)", fontsize=12, fontweight="bold")

    for ax, model in zip(axes, top_models):
        data = [sub[(sub["encoding"] == enc) & (sub["model"] == model)]["mae"].values
                for enc in ENC_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, enc in zip(bp["boxes"], ENC_ORDER):
            patch.set_facecolor(ENC_COLORS[enc])
            patch.set_alpha(0.75)
        ax.set_xticklabels(["Struct\nonly", "Struct\n+LLM", "HF\nonly", "HF\n+LLM"],
                            fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
        ax.set_ylabel("MAE (min)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig3_fold_boxplots")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Actual vs Predicted scatter (best model: LightGBM + HF_LLM)
# ─────────────────────────────────────────────────────────────────────────────
def fig_actual_vs_predicted(df_pred):
    sub = df_pred[(df_pred["encoding"] == "huggingface_llm") &
                  (df_pred["model"] == "lightgbm")].copy()

    # sample for readability
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=42)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub["actual"], sub["predicted"], alpha=0.15, s=8,
               color="#2980B9", rasterized=True, label="Cases")

    lim = max(sub["actual"].max(), sub["predicted"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")

    # regression line
    slope, intercept, r, p, _ = stats.linregress(sub["actual"], sub["predicted"])
    xs = np.linspace(0, lim, 200)
    ax.plot(xs, slope * xs + intercept, "#E67E22", linewidth=1.5,
            label=f"Regression (R={r:.3f})")

    ax.set_xlabel("Actual Case Duration (min)", fontsize=11)
    ax.set_ylabel("Predicted Case Duration (min)", fontsize=11)
    ax.set_title("Actual vs. Predicted — LightGBM + HuggingFace + LLM Features\n(5,000 random validation cases)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9)

    # stats box
    mae  = np.mean(np.abs(sub["actual"] - sub["predicted"]))
    rmse = np.sqrt(np.mean((sub["actual"] - sub["predicted"])**2))
    textstr = f"MAE  = {mae:.1f} min\nRMSE = {rmse:.1f} min\nR²   = {r**2:.3f}"
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    savefig(fig, "fig4_actual_vs_predicted")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Residual / Error distribution (best model)
# ─────────────────────────────────────────────────────────────────────────────
def fig_residuals(df_pred):
    sub = df_pred[(df_pred["encoding"] == "huggingface_llm") &
                  (df_pred["model"] == "lightgbm")].copy()
    sub["error"] = sub["predicted"] - sub["actual"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Prediction Error Analysis — LightGBM + HF + LLM Features", fontsize=12, fontweight="bold")

    # Histogram
    ax = axes[0]
    ax.hist(sub["error"], bins=80, color="#2980B9", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(sub["error"].mean(), color="#E67E22", linestyle="-", linewidth=1.5,
               label=f"Mean = {sub['error'].mean():.1f} min")
    ax.set_xlabel("Prediction Error (min)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Error vs Actual (heteroskedasticity check)
    ax = axes[1]
    if len(sub) > 5000:
        s = sub.sample(5000, random_state=1)
    else:
        s = sub
    ax.scatter(s["actual"], s["error"], alpha=0.1, s=6, color="#8E44AD", rasterized=True)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Actual Duration (min)", fontsize=11)
    ax.set_ylabel("Prediction Error (min)", fontsize=11)
    ax.set_title("Error vs. Actual Duration", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig5_residuals")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Encoding ablation: incremental benefit of adding embeddings/LLM
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation(df):
    top_models = ["xgboost", "lightgbm"]
    agg = (df[df["model"].isin(top_models)]
           .groupby(["encoding", "model"])
           .agg(mae_mean=("mae","mean"), mae_std=("mae","std"),
                r2_mean=("r2","mean"), r2_std=("r2","std"))
           .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ablation Study: Incremental Benefit of Embedding and LLM Features\n(XGBoost & LightGBM)", fontsize=12, fontweight="bold")

    x = np.arange(len(ENC_ORDER))
    w = 0.35

    for ax, metric, ylabel, best_fn in [
        (axes[0], "mae", "MAE (min) ↓ lower is better", np.argmin),
        (axes[1], "r2",  "R² Score ↑ higher is better", np.argmax),
    ]:
        for i, model in enumerate(top_models):
            sub  = agg[agg["model"] == model].set_index("encoding")
            means = [sub.loc[e, f"{metric}_mean"] if e in sub.index else np.nan for e in ENC_ORDER]
            stds  = [sub.loc[e, f"{metric}_std"]  if e in sub.index else 0      for e in ENC_ORDER]
            offset = (i - 0.5) * w
            ax.bar(x + offset, means, w, yerr=stds, label=MODEL_LABELS[model],
                   alpha=0.8, capsize=4,
                   color=["#2980B9", "#27AE60"][i])

        ax.set_xticks(x)
        ax.set_xticklabels([ENC_LABELS[e].replace(" (", "\n(") for e in ENC_ORDER], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # annotate improvement arrows
        if metric == "mae":
            ax.annotate("", xy=(x[-1], agg[agg["model"]=="lightgbm"].set_index("encoding").loc["huggingface_llm","mae_mean"]),
                        xytext=(x[0], agg[agg["model"]=="lightgbm"].set_index("encoding").loc["only_structured","mae_mean"]),
                        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5))

    fig.tight_layout()
    savefig(fig, "fig6_ablation")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Training time comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_training_time(df):
    agg = df.groupby(["encoding", "model"])["train_time_s"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(MODEL_ORDER))
    w = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for enc, off in zip(ENC_ORDER, offsets):
        sub = agg[agg["encoding"] == enc].set_index("model")
        vals = [sub.loc[m, "train_time_s"] if m in sub.index else 0 for m in MODEL_ORDER]
        ax.bar(x + off * w, vals, w, label=ENC_LABELS[enc],
               color=ENC_COLORS[enc], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
    ax.set_ylabel("Mean Training Time (seconds)", fontsize=11)
    ax.set_title("Training Time per Model and Encoding\n(mean across 5 folds, includes Optuna HPO)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig7_training_time")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Summary radar chart (top 4 models)
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar(df):
    top = [
        ("huggingface_llm", "lightgbm"),
        ("huggingface_llm", "xgboost"),
        ("huggingface",     "lightgbm"),
        ("only_structured", "lightgbm"),
    ]
    metrics_cols = ["mae", "rmse", "smape", "r2"]
    metric_labels = ["MAE (↓)", "RMSE (↓)", "SMAPE (↓)", "R² (↑)"]

    agg = df.groupby(["encoding","model"])[metrics_cols].mean()

    # Normalise: 0=worst, 1=best for each metric
    col_min = agg[metrics_cols].min()
    col_max = agg[metrics_cols].max()
    norm = (agg[metrics_cols] - col_min) / (col_max - col_min)
    # Invert lower-is-better
    for c in ["mae","rmse","smape"]:
        norm[c] = 1 - norm[c]

    N = len(metrics_cols)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ["#27AE60", "#2980B9", "#E67E22", "#7F8C8D"]

    for (enc, mdl), color in zip(top, colors):
        vals = norm.loc[(enc, mdl), metrics_cols].tolist()
        vals += vals[:1]
        label = f"{ENC_LABELS[enc].split('(')[0].strip()}\n+ {MODEL_LABELS[mdl]}"
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title("Model Comparison Radar\n(normalised, higher = better)", fontsize=11,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig8_radar")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data …")
    df_metrics = load_metrics()
    df_preds   = load_predictions()

    print("Generating figures …")
    fig_heatmap(df_metrics)
    fig_grouped_bar(df_metrics)
    fig_fold_boxplots(df_metrics)
    fig_actual_vs_predicted(df_preds)
    fig_residuals(df_preds)
    fig_ablation(df_metrics)
    fig_training_time(df_metrics)
    fig_radar(df_metrics)

    print(f"\n[OK] All figures saved to {OUT_DIR}/")
