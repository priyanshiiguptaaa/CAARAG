"""
step5_evaluation.py
────────────────────────────────────────────────────────────────────────────────
STEP 5 — Final Evaluation, Calibration Analysis & Visualization
────────────────────────────────────────────────────────────────────────────────

This script produces everything needed for the journal Results section:

    5.1  Answer Quality       — EM, F1, ROUGE-L (Table 1 in paper)
    5.2  Calibration Analysis — ECE, Reliability Diagram (Figure 4)
    5.3  Adaptation Analysis  — Lift, Convergence Rate (Table 2)
    5.4  Visualization        — Matplotlib plots saved to data/plots/

Run:
    python step5_evaluation.py

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.logger import logger
from utils.config_loader import load_config
from evaluation.answer_metrics import evaluate_batch, exact_match, token_f1, rouge_l
from evaluation.calibration import compute_ece, confidence_accuracy_correlation, split_by_confidence
from evaluation.adaptation_analysis import analyse_adaptation


# ════════════════════════════════════════════════════════════════════════════
# 5.0  Load Data
# ════════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# 5.4  Visualization (Matplotlib — no external deps beyond numpy)
# ════════════════════════════════════════════════════════════════════════════

def save_plots(
    eval_data: Dict,
    calibration_data: Dict,
    adaptation_data: Dict,
    per_query: List[Dict],
    confidences: List[float],
    f1_scores: List[float],
    plots_dir: str,
):
    """Generate all publication-quality plots and save to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plots. Run: pip install matplotlib")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # ── Style ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#e6edf3",
        "text.color":       "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "grid.color":       "#21262d",
        "grid.alpha":       0.6,
        "font.family":      "sans-serif",
        "font.size":        11,
    })

    # ── FIGURE 1: Answer Quality Bar Chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics  = ["Exact Match", "Token F1", "ROUGE-L F1"]
    values   = [
        eval_data["avg_em"],
        eval_data["avg_f1"],
        eval_data["avg_rouge_l_f1"],
    ]
    colors = ["#6366f1", "#34d399", "#38bdf8"]
    bars = ax.bar(metrics, values, color=colors, width=0.55, edgecolor="#30363d", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=12, color="#e6edf3")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Answer Quality Metrics", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig1_answer_quality.png"), dpi=200)
    plt.close()
    logger.success(f"  → fig1_answer_quality.png")

    # ── FIGURE 2: Reliability Diagram (ECE) ───────────────────────────────
    bin_data = calibration_data.get("bin_data", [])
    if bin_data:
        fig, ax = plt.subplots(figsize=(7, 6))
        bin_mids   = [b["bin_mid"] for b in bin_data]
        bin_accs   = [b["avg_accuracy"] for b in bin_data]
        bin_confs  = [b["avg_confidence"] for b in bin_data]
        bin_counts = [b["count"] for b in bin_data]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "--", color="#f85149", linewidth=1.5, label="Perfect Calibration", alpha=0.7)
        
        # Bar chart of actual accuracy per bin
        bar_width = 0.08
        bars = ax.bar(bin_mids, bin_accs, width=bar_width, color="#6366f1", edgecolor="#30363d",
                       alpha=0.85, label="Actual Accuracy", zorder=3)
        
        # Overlay gap shading
        for mid, acc, conf in zip(bin_mids, bin_accs, bin_confs):
            if conf > 0:
                ax.plot([mid, mid], [acc, conf], color="#fbbf24", linewidth=2, alpha=0.6)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction of Correct Answers")
        ece_val = calibration_data["ece"]
        ax.set_title(f"Reliability Diagram  (ECE = {ece_val:.4f})", fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="upper left", framealpha=0.3)
        ax.grid(linestyle="--", alpha=0.3)

        # Secondary axis: sample counts
        ax2 = ax.twinx()
        ax2.bar(bin_mids, bin_counts, width=bar_width * 0.6, color="#34d399", alpha=0.3, label="Sample Count")
        ax2.set_ylabel("Sample Count", color="#34d399")
        ax2.tick_params(axis="y", labelcolor="#34d399")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig2_reliability_diagram.png"), dpi=200)
        plt.close()
        logger.success(f"  → fig2_reliability_diagram.png")

    # ── FIGURE 3: Confidence vs F1 Scatter ────────────────────────────────
    if confidences and f1_scores:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(confidences, f1_scores, c="#6366f1", s=70, alpha=0.75, edgecolors="#30363d", linewidth=0.5, zorder=3)
        
        # Trend line
        if len(confidences) > 2:
            z = np.polyfit(confidences, f1_scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(confidences), max(confidences), 50)
            ax.plot(x_line, p(x_line), "--", color="#fbbf24", linewidth=2, alpha=0.8, label=f"Trend (r={eval_data.get('correlation', {}).get('pearson_r', 0):.3f})")
        
        ax.set_xlabel("Calibrated Confidence (C)")
        ax.set_ylabel("Token F1 Score")
        ax.set_title("Confidence–Accuracy Correlation", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="lower right", framealpha=0.3)
        ax.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig3_confidence_vs_f1.png"), dpi=200)
        plt.close()
        logger.success(f"  → fig3_confidence_vs_f1.png")

    # ── FIGURE 4: Adaptation Effectiveness ────────────────────────────────
    per_round = adaptation_data.get("per_round_signals", {})
    if per_round:
        rounds   = sorted(per_round.keys())
        round_ns = [int(r.split("_")[1]) for r in rounds]
        avg_cs   = [per_round[r]["avg_C"] for r in rounds]
        avg_srs  = [per_round[r]["avg_Sr"] for r in rounds]
        avg_sls  = [per_round[r]["avg_Sl"] for r in rounds]
        avg_scs  = [per_round[r]["avg_Sc"] for r in rounds]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(round_ns, avg_cs,  "o-", color="#6366f1", linewidth=2.5, markersize=8, label="C (Calibrated)", zorder=4)
        ax.plot(round_ns, avg_srs, "s--", color="#f0883e", linewidth=1.5, markersize=6, label="Sᵣ (Retrieval)", alpha=0.7)
        ax.plot(round_ns, avg_sls, "d--", color="#d2a8ff", linewidth=1.5, markersize=6, label="Sₗ (LLM)", alpha=0.7)
        ax.plot(round_ns, avg_scs, "^--", color="#34d399", linewidth=1.5, markersize=6, label="Sc (Consistency)", alpha=0.7)
        
        tau = adaptation_data.get("tau", 0.7)
        ax.axhline(y=tau, color="#f85149", linestyle="--", alpha=0.6, label=f"τ = {tau}")
        
        ax.set_xlabel("Retrieval Round")
        ax.set_ylabel("Average Signal Value")
        ax.set_title("Signal Evolution Across Adaptive Rounds", fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(round_ns)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right", framealpha=0.3, fontsize=9)
        ax.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig4_adaptation_evolution.png"), dpi=200)
        plt.close()
        logger.success(f"  → fig4_adaptation_evolution.png")

    # ── FIGURE 5: Round Distribution ──────────────────────────────────────
    round_dist = adaptation_data.get("round_distribution", {})
    if round_dist:
        fig, ax = plt.subplots(figsize=(6, 4))
        rounds_x = sorted(round_dist.keys())
        counts   = [round_dist[r] for r in rounds_x]
        colors_r = ["#34d399" if r == 1 else "#fbbf24" if r == 2 else "#f87171" for r in rounds_x]
        
        bars = ax.bar([str(r) for r in rounds_x], counts, color=colors_r, edgecolor="#30363d", width=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(c), ha="center", va="bottom", fontweight="bold", fontsize=12)
        
        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel("Query Count")
        ax.set_title("Distribution of Adaptive Rounds", fontsize=14, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig5_round_distribution.png"), dpi=200)
        plt.close()
        logger.success(f"  → fig5_round_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# 5.5  Report Generator
# ════════════════════════════════════════════════════════════════════════════

def print_report(eval_data: Dict, cal_data: Dict, adapt_data: Dict, split_data: Dict):
    """Print a formatted report to the console and logger."""
    
    logger.info("\n" + "═" * 74)
    logger.info("     📊 STEP 5 — FINAL EVALUATION REPORT")
    logger.info("═" * 74)
    
    # ── Table 1: Answer Quality ───────────────────────────────────────────
    logger.info("\n┌─────────────────────────────────────────────────────┐")
    logger.info("│  TABLE 1: Answer Quality Metrics                    │")
    logger.info("├──────────────────────┬──────────────────────────────┤")
    logger.info(f"│  Exact Match (EM)    │  {eval_data['avg_em']:.4f}                        │")
    logger.info(f"│  Token F1            │  {eval_data['avg_f1']:.4f}                        │")
    logger.info(f"│  Precision           │  {eval_data['avg_precision']:.4f}                        │")
    logger.info(f"│  Recall              │  {eval_data['avg_recall']:.4f}                        │")
    logger.info(f"│  ROUGE-L F1          │  {eval_data['avg_rouge_l_f1']:.4f}                        │")
    logger.info(f"│  N (samples)         │  {eval_data['n']}                             │")
    logger.info("└──────────────────────┴──────────────────────────────┘")
    
    # ── Table 2: Calibration ──────────────────────────────────────────────
    logger.info("\n┌─────────────────────────────────────────────────────┐")
    logger.info("│  TABLE 2: Confidence Calibration                    │")
    logger.info("├──────────────────────┬──────────────────────────────┤")
    logger.info(f"│  ECE                 │  {cal_data['ece']:.4f}                        │")
    logger.info(f"│  Pearson r (C vs F1) │  {eval_data.get('correlation', {}).get('pearson_r', 0):.4f}                        │")
    logger.info(f"│  High-Conf Avg F1    │  {split_data.get('high_conf_avg_f1', 0):.4f}  (n={split_data.get('high_conf_count', 0)})               │")
    logger.info(f"│  Low-Conf  Avg F1    │  {split_data.get('low_conf_avg_f1', 0):.4f}  (n={split_data.get('low_conf_count', 0)})               │")
    logger.info(f"│  ΔF1 (high - low)    │  {split_data.get('delta_f1', 0):.4f}                        │")
    logger.info("└──────────────────────┴──────────────────────────────┘")
    
    # ── Table 3: Adaptation Effectiveness ─────────────────────────────────
    logger.info("\n┌─────────────────────────────────────────────────────┐")
    logger.info("│  TABLE 3: Adaptive Retrieval Effectiveness          │")
    logger.info("├──────────────────────┬──────────────────────────────┤")
    logger.info(f"│  Adaptation Rate     │  {adapt_data.get('adaptation_rate', 0)*100:.1f}%                         │")
    logger.info(f"│  Avg Rounds          │  {adapt_data.get('avg_rounds', 0):.2f}                         │")
    logger.info(f"│  Avg Round-1 Conf    │  {adapt_data.get('avg_round1_confidence', 0):.4f}                        │")
    logger.info(f"│  Avg Final Conf      │  {adapt_data.get('avg_final_confidence', 0):.4f}                        │")
    logger.info(f"│  Avg Conf Lift       │  {adapt_data.get('avg_confidence_lift', 0):+.4f}                        │")
    logger.info(f"│  Convergence Rate    │  {adapt_data.get('convergence_rate', 0)*100:.1f}% ({adapt_data.get('queries_rescued', 0)}/{adapt_data.get('queries_started_below_tau', 0)})                  │")
    logger.info("└──────────────────────┴──────────────────────────────┘")
    
    # ── Per-Round breakdown ───────────────────────────────────────────────
    per_round = adapt_data.get("per_round_signals", {})
    if per_round:
        logger.info("\n  Per-Round Signal Averages:")
        for rnd, data in sorted(per_round.items()):
            logger.info(
                f"    {rnd}: n={data['count']:2d}  "
                f"Sr={data['avg_Sr']:.3f}  Sl={data['avg_Sl']:.3f}  "
                f"Sc={data['avg_Sc']:.3f}  C={data['avg_C']:.3f}"
            )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 80)
    logger.info("  STEP 5 — Final Evaluation & Confidence Calibration Analysis ")
    logger.info("=" * 80)

    # ── Load Config & Data ────────────────────────────────────────────────
    cfg      = load_config(str(ROOT / "configs" / "config.yaml"))
    e_cfg    = cfg.get("evaluation", {})
    a_cfg    = cfg.get("adaptive", {})
    
    adaptive_path = a_cfg.get("results_path", "data/adaptive_results.json")
    results_path  = e_cfg.get("results_path", "data/eval_results.json")
    plots_dir     = e_cfg.get("plots_dir", "data/plots")
    cal_bins      = e_cfg.get("calibration_bins", 10)
    tau           = a_cfg.get("threshold_tau", 0.7)

    logger.info(f"Loading results from: {adaptive_path}")
    results = load_results(adaptive_path)
    logger.success(f"Loaded {len(results)} query results.")

    # ── 5.1  Answer Quality Metrics ───────────────────────────────────────
    logger.info("\n── 5.1 Computing Answer Quality (EM, F1, ROUGE-L) ──")
    predictions = [r["final_answer"] for r in results]
    golds       = [r["gold_answer"] for r in results]
    
    eval_data = evaluate_batch(predictions, golds)
    logger.success(f"  EM={eval_data['avg_em']:.4f}  F1={eval_data['avg_f1']:.4f}  ROUGE-L={eval_data['avg_rouge_l_f1']:.4f}")

    # ── 5.2  Calibration Analysis ─────────────────────────────────────────
    logger.info("\n── 5.2 Computing Calibration Analysis (ECE) ──")
    confidences = [r["final_confidence"] for r in results]
    f1_scores   = [pq["f1"] for pq in eval_data["per_query"]]
    
    # Correctness label: F1 > 0.5 → correct for ECE
    correctness = [1.0 if f > 0.5 else 0.0 for f in f1_scores]
    
    cal_data = compute_ece(confidences, correctness, n_bins=cal_bins)
    logger.success(f"  ECE = {cal_data['ece']:.4f}")
    
    # Correlation
    corr_data = confidence_accuracy_correlation(confidences, f1_scores)
    eval_data["correlation"] = corr_data
    logger.info(f"  Pearson r (Confidence vs F1) = {corr_data['pearson_r']:.4f}")
    
    # High/Low split
    split_data = split_by_confidence(confidences, f1_scores, threshold=tau)
    logger.info(f"  High-Conf F1 = {split_data['high_conf_avg_f1']:.4f}  |  Low-Conf F1 = {split_data['low_conf_avg_f1']:.4f}")

    # ── 5.3  Adaptation Effectiveness ─────────────────────────────────────
    logger.info("\n── 5.3 Analysing Adaptation Effectiveness ──")
    adapt_data = analyse_adaptation(results, tau=tau)
    logger.success(f"  Adaptation Rate: {adapt_data['adaptation_rate']*100:.1f}%")
    logger.info(f"  Convergence Rate: {adapt_data['convergence_rate']*100:.1f}%")
    logger.info(f"  Avg Confidence Lift: {adapt_data['avg_confidence_lift']:+.4f}")

    # ── 5.4  Visualization ────────────────────────────────────────────────
    logger.info(f"\n── 5.4 Generating Plots → {plots_dir}/ ──")
    save_plots(
        eval_data=eval_data,
        calibration_data=cal_data,
        adaptation_data=adapt_data,
        per_query=eval_data["per_query"],
        confidences=confidences,
        f1_scores=f1_scores,
        plots_dir=plots_dir,
    )

    # ── 5.5  Print Report ─────────────────────────────────────────────────
    print_report(eval_data, cal_data, adapt_data, split_data)

    # ── 5.6  Save Full Results ────────────────────────────────────────────
    full_results = {
        "answer_quality": {k: v for k, v in eval_data.items() if k != "per_query"},
        "per_query_scores": eval_data["per_query"],
        "calibration": cal_data,
        "correlation": corr_data,
        "confidence_split": split_data,
        "adaptation": adapt_data,
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
    logger.success(f"\nFull evaluation saved → {results_path}")

    logger.info("\n" + "═" * 80)
    logger.info(" ✅ STEP 5 COMPLETE — All evaluation metrics computed.")
    logger.info(" Your journal results section is ready!")
    logger.info("")
    logger.info(" Key Files:")
    logger.info(f"   📊 {results_path}           — Full JSON report")
    logger.info(f"   📈 {plots_dir}/fig1_answer_quality.png     — Answer bar chart")
    logger.info(f"   📈 {plots_dir}/fig2_reliability_diagram.png — ECE diagram")
    logger.info(f"   📈 {plots_dir}/fig3_confidence_vs_f1.png    — Correlation scatter")
    logger.info(f"   📈 {plots_dir}/fig4_adaptation_evolution.png — Signal evolution")
    logger.info(f"   📈 {plots_dir}/fig5_round_distribution.png  — Round histogram")
    logger.info("═" * 80)


if __name__ == "__main__":
    main()
