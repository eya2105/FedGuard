"""
visualize.py  —  Plotting training curves and comparing FL algorithms
=====================================================================
Generates the four plots you need for your project report:
  1. Training loss curves per client across FL rounds
  2. Per-client F1 scores: FedAvg vs FedProx vs FedNova
  3. FedProx μ sweep — how regularisation strength affects performance
  4. Confusion matrix heatmap per client
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
from sklearn.metrics import ConfusionMatrixDisplay


# ── consistent colour palette across all plots ────────────────────────────────
CLIENT_COLORS = {
    "ctu13":  "#2E75B6",
    "suee8":  "#C0392B",
    "unsw":   "#27AE60",
    "caida":  "#8E44AD",
}
ALGO_COLORS = {
    "fedavg":  "#2E75B6",
    "fedprox": "#E67E22",
    "fednova": "#27AE60",
}


def plot_training_curves(
    history_fedavg:  Dict,
    history_fedprox: Dict,
    save_path: str = "training_curves.png",
):
    """
    Plots round-level mean training loss for FedAvg vs FedProx side by side.

    What to look for in this plot:
    --------------------------------
    - FedProx should converge more smoothly (less oscillation) because the
      proximal term prevents clients from taking large inconsistent steps.
    - If both curves look identical, μ is too small (clients behave like FedAvg).
    - If FedProx converges slower, μ is too large (clients barely adapt locally).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    rounds_avg  = range(1, len(history_fedavg["round_train_losses"])  + 1)
    rounds_prox = range(1, len(history_fedprox["round_train_losses"]) + 1)

    ax1.plot(rounds_avg,  history_fedavg["round_train_losses"],
             color=ALGO_COLORS["fedavg"],  lw=2, label="FedAvg")
    ax1.plot(rounds_prox, history_fedprox["round_train_losses"],
             color=ALGO_COLORS["fedprox"], lw=2, label="FedProx")
    ax1.set_xlabel("FL Round")
    ax1.set_ylabel("Mean training loss (across clients)")
    ax1.set_title("Training loss convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot per-client F1 trajectories if intermediate evaluations exist
    ax2.set_xlabel("FL Round")
    ax2.set_ylabel("F1 score (attack class)")
    ax2.set_title("Per-client F1 during training")

    for algo_name, history, style in [
        ("FedAvg",  history_fedavg,  "-"),
        ("FedProx", history_fedprox, "--"),
    ]:
        for rnd, client_metrics in history["per_client_metrics"].items():
            for client_name, metrics in client_metrics.items():
                color = CLIENT_COLORS.get(client_name, "gray")
                ax2.plot(
                    int(rnd), metrics["f1"],
                    marker="o", color=color, linestyle=style,
                    label=f"{algo_name} · {client_name}" if int(rnd) == min(
                        history["per_client_metrics"].keys()
                    ) else ""
                )

    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_algorithm_comparison(
    results: Dict[str, Dict[str, Dict]],
    metric:  str = "f1",
    save_path: str = "algorithm_comparison.png",
):
    """
    Bar chart comparing FedAvg, FedProx, and FedNova on each dataset.

    results format: { "fedavg": { "ctu13": metrics, ... }, "fedprox": {...}, ... }

    What to look for:
    -----------------
    - Is FedProx more uniform across clients (smaller spread between the
      tallest and shortest bars)?
    - Does FedNova close the gap on SUEE-8 or CAIDA — the clients that
      struggle most with FedAvg due to step-count or size imbalance?
    - Is the recall for the attack class consistently high (>0.90)?
      If recall is low on any client, the system is missing attacks there.
    """
    client_names = list(next(iter(results.values())).keys())
    algo_names   = list(results.keys())
    n_algos      = len(algo_names)
    n_clients    = len(client_names)

    x = np.arange(n_clients)
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, algo in enumerate(algo_names):
        values = [
            results[algo][client].get(metric, 0)
            for client in client_names
        ]
        bars = ax.bar(
            x + i * width, values, width,
            label=algo.upper(),
            color=ALGO_COLORS.get(algo, "gray"),
            alpha=0.85,
        )
        # Print value on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.upper() for c in client_names])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} comparison: FedAvg vs FedProx vs FedNova")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mu_sweep(
    sweep_results: Dict[float, Dict[str, Dict]],
    metric: str = "f1",
    save_path: str = "mu_sweep.png",
):
    """
    Line plot showing how F1 (or another metric) changes with FedProx μ
    for each client dataset.

    What to look for:
    -----------------
    - The μ value where all four client curves are simultaneously high
      (minimum gap between best and worst client) is the optimal μ.
    - If one client's performance peaks at a different μ than the others,
      it signals that this client's data is particularly different from
      the federation's average — worth documenting in your report.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    mu_values    = sorted(sweep_results.keys())
    client_names = list(next(iter(sweep_results.values())).keys())

    for client in client_names:
        values = [sweep_results[mu][client][metric] for mu in mu_values]
        ax.plot(
            [str(m) for m in mu_values], values,
            marker="o", lw=2, label=client.upper(),
            color=CLIENT_COLORS.get(client, "gray"),
        )

    ax.set_xlabel("FedProx μ value")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"FedProx μ sweep — effect on per-client {metric.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(
    per_client_metrics: Dict[str, Dict],
    save_path: str = "confusion_matrices.png",
):
    """
    4-panel confusion matrix grid, one per client.

    The confusion matrix shows you exactly where the model is making errors.
    For intrusion detection the critical cells are:
      - FN (bottom-left): real attacks called normal → missed intrusions
      - FP (top-right):   normal called attack → false alarms

    In your report, highlight that the FN count is the primary concern for
    a security system.  A system that misses 1% of attacks is dangerous
    even if it has very low false alarms.
    """
    n = len(per_client_metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (client_name, metrics) in zip(axes, per_client_metrics.items()):
        cm = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Attack"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{client_name.upper()}\n"
                     f"F1={metrics['f1']:.3f}  "
                     f"FNR={metrics['false_neg_rate']:.3f}")

    plt.suptitle("Confusion matrices — global model per client", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
