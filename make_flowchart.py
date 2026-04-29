"""
make_flowchart.py
Generates a Graphviz pipeline flowchart as both .dot and .pdf/.png.
Run: python make_flowchart.py
Requires: graphviz (pip install graphviz) + Graphviz system install
"""

import os
from graphviz import Digraph

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def build_pipeline_flowchart():
    dot = Digraph(
        "Clinical_Surgery_Pipeline",
        comment="Clinical Surgery Case-Time Prediction Pipeline",
        format="pdf",
    )
    dot.attr(
        rankdir="TB",
        size="10,16",
        dpi="300",
        fontname="Helvetica",
        splines="ortho",
        nodesep="0.6",
        ranksep="0.7",
        bgcolor="white",
    )

    # ── node defaults ────────────────────────────────────────────────────────
    dot.attr("node", fontname="Helvetica", fontsize="11", margin="0.15,0.10")
    dot.attr("edge", fontname="Helvetica", fontsize="9", color="#444444")

    # ── INPUT ────────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_input") as c:
        c.attr(label="Input Data", style="filled", fillcolor="#EAF4FB",
               color="#2980B9", fontname="Helvetica Bold", fontsize="12")
        c.node("csv", "casetime.csv\n(180,370 surgical cases)",
               shape="cylinder", style="filled", fillcolor="#D6EAF8", color="#2980B9")

    # ── STAGE 01 ─────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_s01") as c:
        c.attr(label="Stage 01 — Data Cleaning", style="filled",
               fillcolor="#EAF9F1", color="#27AE60", fontname="Helvetica Bold", fontsize="12")
        c.node("s01", "Data Cleaning\n• Remove outliers (IQR)\n• Impute missing values\n• Normalise timestamps",
               shape="box", style="filled,rounded", fillcolor="#D5F5E3", color="#27AE60")
        c.node("db", "surgical_data.db\n(SQLite, cleaned)",
               shape="cylinder", style="filled", fillcolor="#D5F5E3", color="#27AE60")

    # ── STAGE 02 ─────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_s02") as c:
        c.attr(label="Stage 02 — Text Embeddings", style="filled",
               fillcolor="#FEF9E7", color="#F39C12", fontname="Helvetica Bold", fontsize="12")
        c.node("hf", "HuggingFace (local)\nall-MiniLM-L6-v2\n384-d vectors",
               shape="box", style="filled,rounded", fillcolor="#FCF3CF", color="#F39C12")
        c.node("gem", "Google Gemini API\ntext-embedding-004\n768-d → PCA 384-d",
               shape="box", style="filled,rounded,dashed", fillcolor="#FCF3CF", color="#E67E22")
        c.node("emb_cache", "embed_cache/\n(.npy arrays)",
               shape="folder", style="filled", fillcolor="#FCF3CF", color="#F39C12")

    # ── STAGE 02b ────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_s02b") as c:
        c.attr(label="Stage 02b — LLM Feature Extraction", style="filled",
               fillcolor="#F5EEF8", color="#8E44AD", fontname="Helvetica Bold", fontsize="12")
        c.node("groq", "Groq API\nLlama-3.1-8b-instant\n(1,730 unique procedures)",
               shape="box", style="filled,rounded", fillcolor="#E8DAEF", color="#8E44AD")
        c.node("llm_feat", "LLM Features (per procedure)\n• body_region (10-class)\n• complexity (1–5)\n• is_laparoscopic / is_robotic\n• is_bilateral / is_emergency\n• n_procedures",
               shape="note", style="filled", fillcolor="#E8DAEF", color="#8E44AD")

    # ── STAGE 03 ─────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_s03") as c:
        c.attr(label="Stage 03 — Feature Engineering", style="filled",
               fillcolor="#FDFEFE", color="#2C3E50", fontname="Helvetica Bold", fontsize="12")
        c.node("s03", "5-Fold Cross-Validation\n• Mean imputation\n• One-hot encoding (surgeon, room, …)\n• PCA on Gemini embeddings (768→384)\n• 4 encoding variants assembled",
               shape="box", style="filled,rounded", fillcolor="#F2F3F4", color="#2C3E50")
        c.node("fold_db", "fold_encoded.db\n(3.3 GB, 5 folds × 4 encodings)",
               shape="cylinder", style="filled", fillcolor="#F2F3F4", color="#2C3E50")

    # ── STAGE 04 ─────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_s04") as c:
        c.attr(label="Stage 04 — Model Training & Evaluation", style="filled",
               fillcolor="#FDEDEC", color="#C0392B", fontname="Helvetica Bold", fontsize="12")
        c.node("optuna", "Optuna HPO\n(5 trials, 10% tune set\nTPE sampler)",
               shape="ellipse", style="filled", fillcolor="#FADBD8", color="#C0392B")
        c.node("models", "7 Models\nRidge · Lasso · ElasticNet\nRandom Forest · XGBoost\nLightGBM · MLP",
               shape="box", style="filled,rounded", fillcolor="#FADBD8", color="#C0392B")
        c.node("eval", "5-Fold Evaluation\nMAE · RMSE · R² · SMAPE",
               shape="box", style="filled,rounded", fillcolor="#FADBD8", color="#C0392B")

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_out") as c:
        c.attr(label="Outputs", style="filled", fillcolor="#EAF4FB",
               color="#1A5276", fontname="Helvetica Bold", fontsize="12")
        c.node("result_db", "result.db\n• metrics (140 rows)\n• predictions\n• feature_importance\n• hyperparameters",
               shape="cylinder", style="filled", fillcolor="#D6EAF8", color="#1A5276")
        c.node("best", "Best Model\nhuggingface_llm + LightGBM\nMAE=26.58 min  R²=0.852",
               shape="star", style="filled", fillcolor="#AED6F1", color="#1A5276",
               fontname="Helvetica Bold")

    # ── ENCODING LEGEND ──────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="Encoding Variants (Stage 03 → 04)", style="dashed",
               color="#7F8C8D", fontname="Helvetica", fontsize="10")
        c.node("enc_table",
               """<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
  <TR><TD BGCOLOR="#D5D8DC"><B>Encoding</B></TD><TD BGCOLOR="#D5D8DC"><B>Features</B></TD></TR>
  <TR><TD>only_structured</TD><TD>38 tabular</TD></TR>
  <TR><TD>only_llm</TD><TD>38 + 16 LLM = 54</TD></TR>
  <TR><TD>huggingface</TD><TD>38 + 384 HF = 422</TD></TR>
  <TR><TD>huggingface_llm</TD><TD>38 + 384 HF + 16 LLM = 438</TD></TR>
</TABLE>>""",
               shape="none", margin="0")

    # ── EDGES ────────────────────────────────────────────────────────────────
    dot.edge("csv", "s01")
    dot.edge("s01", "db")
    dot.edge("db", "hf", label=" procedure\ntext")
    dot.edge("db", "gem", label=" procedure\ntext", style="dashed", color="#E67E22")
    dot.edge("db", "groq", label=" procedure\ntext")
    dot.edge("hf", "emb_cache")
    dot.edge("gem", "emb_cache", style="dashed", color="#E67E22")
    dot.edge("groq", "llm_feat")
    dot.edge("db", "s03", label=" structured\nfeatures")
    dot.edge("emb_cache", "s03", label=" embeddings")
    dot.edge("llm_feat", "s03", label=" LLM\nfeatures")
    dot.edge("s03", "fold_db")
    dot.edge("fold_db", "optuna")
    dot.edge("optuna", "models", label=" best\nparams")
    dot.edge("models", "eval")
    dot.edge("eval", "result_db")
    dot.edge("result_db", "best")

    out = os.path.join(OUT_DIR, "pipeline_flowchart")
    dot.render(out, cleanup=True)
    print(f"[OK] PDF saved -> {out}.pdf")

    # Also save PNG
    dot2 = dot.copy()
    dot2.format = "png"
    dot2.attr(dpi="200")
    out2 = os.path.join(OUT_DIR, "pipeline_flowchart")
    dot2.render(out2, cleanup=True)
    print(f"[OK] PNG saved -> {out2}.png")

    # Save .dot source
    dot_src = out + ".dot"
    dot.save(dot_src)
    print(f"[OK] DOT source -> {dot_src}")

    return out


if __name__ == "__main__":
    build_pipeline_flowchart()
