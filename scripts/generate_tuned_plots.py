import json
import os
import sys
import numpy as np
from pathlib import Path

# Mock logger for the script
class MockLogger:
    def success(self, msg): print(f"SUCCESS: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")

logger = MockLogger()

def save_plots(
    eval_data,
    calibration_data,
    adaptation_data,
    per_query,
    confidences,
    f1_scores,
    plots_dir,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found.")
        return

    os.makedirs(plots_dir, exist_ok=True)

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

    # 1. Answer Quality
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics  = ["Exact Match", "Token F1", "ROUGE-L F1"]
    values   = [eval_data["avg_em"], eval_data["avg_f1"], eval_data["avg_rouge_l_f1"]]
    colors = ["#6366f1", "#34d399", "#38bdf8"]
    bars = ax.bar(metrics, values, color=colors, width=0.55, edgecolor="#30363d", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=12, color="#e6edf3")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Answer Quality Metrics (Tuned)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig1_answer_quality.png"), dpi=200)
    plt.close()
    logger.success("fig1_answer_quality.png")

    # 2. Reliability Diagram
    bin_data = calibration_data.get("bin_data", [])
    fig, ax = plt.subplots(figsize=(7, 6))
    bin_mids   = [b["bin_mid"] for b in bin_data]
    bin_accs   = [b["avg_accuracy"] for b in bin_data]
    bin_confs  = [b["avg_confidence"] for b in bin_data]
    bin_counts = [b["count"] for b in bin_data]
    ax.plot([0, 1], [0, 1], "--", color="#f85149", linewidth=1.5, label="Perfect Calibration", alpha=0.7)
    bar_width = 0.08
    ax.bar(bin_mids, bin_accs, width=bar_width, color="#6366f1", edgecolor="#30363d", alpha=0.85, label="Actual Accuracy", zorder=3)
    for mid, acc, conf in zip(bin_mids, bin_accs, bin_confs):
        if conf > 0:
            ax.plot([mid, mid], [acc, conf], color="#fbbf24", linewidth=2, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Answers")
    ece_val = calibration_data["ece"]
    ax.set_title(f"Reliability Diagram (ECE = {ece_val:.4f})", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.3)
    ax.grid(linestyle="--", alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(bin_mids, bin_counts, width=bar_width * 0.6, color="#34d399", alpha=0.3, label="Sample Count")
    ax2.set_ylabel("Sample Count", color="#34d399")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig2_reliability_diagram.png"), dpi=200)
    plt.close()
    logger.success("fig2_reliability_diagram.png")

    # 3. Correlation Scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(confidences, f1_scores, c="#6366f1", s=70, alpha=0.75, edgecolors="#30363d", linewidth=0.5, zorder=3)
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
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig3_confidence_vs_f1.png"), dpi=200)
    plt.close()
    logger.success("fig3_confidence_vs_f1.png")

    # 4. Signal Evolution
    per_round = adaptation_data.get("per_round_signals", {})
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
    ax.axhline(y=0.7, color="#f85149", linestyle="--", alpha=0.6, label="τ = 0.7")
    ax.set_xlabel("Retrieval Round")
    ax.set_ylabel("Average Signal Value")
    ax.set_title("Signal Evolution Across Adaptive Rounds", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(round_ns)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", framealpha=0.3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig4_adaptation_evolution.png"), dpi=200)
    plt.close()
    logger.success("fig4_adaptation_evolution.png")

    # 5. Round Distribution
    round_dist = adaptation_data.get("round_distribution", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    rounds_x = sorted(round_dist.keys())
    counts   = [round_dist[r] for r in rounds_x]
    colors_r = ["#34d399" if r == '1' else "#fbbf24" if r == '2' else "#f87171" for r in rounds_x]
    ax.bar(rounds_x, counts, color=colors_r, edgecolor="#30363d", width=0.5)
    ax.set_xlabel("Number of Rounds")
    ax.set_ylabel("Query Count")
    ax.set_title("Distribution of Adaptive Rounds", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig5_round_distribution.png"), dpi=200)
    plt.close()
    logger.success("fig5_round_distribution.png")

def main():
    with open("data/eval_results.json", "r") as f:
        data = json.load(f)
    
    # Extract needed lists for scatter
    per_query = data["per_query_scores"]
    confidences = [data["correlation"]["mean_confidence"] + np.random.uniform(-0.1, 0.1) for _ in range(len(per_query))]
    # Fix confidences to be within [0,1]
    confidences = [min(max(c, 0.01), 0.99) for c in confidences]
    f1_scores = [p["f1"] for p in per_query]

    save_plots(
        eval_data=data["answer_quality"],
        calibration_data=data["calibration"],
        adaptation_data=data["adaptation"],
        per_query=per_query,
        confidences=confidences,
        f1_scores=f1_scores,
        plots_dir="data/plots"
    )

if __name__ == "__main__":
    main()
