"""
run_experiment.py  —  Complete experiment pipeline
===================================================
This is the single entry point you run to reproduce all results.

Usage
-----
    python run_experiment.py

Before running:
    1.  Put your four Windower-processed CSV files in the data/ folder.
    2.  Update the DATASET_PATHS dict below to point to your files.
    3.  Check FEATURE_COLUMNS in data.py match your CSV column names exactly.
    4.  If you have a GPU, set DEVICE = "cuda" below.

What this script does:
    Step 1  Load and preprocess all four client datasets.
    Step 2  Train a centralised baseline (all data pooled, no FL).
            This is your upper-bound reference for comparison.
    Step 3  Train with FedAvg.
    Step 4  Train with FedProx (default μ=0.01).
    Step 5  Train with FedNova.
    Step 6  Run a FedProx μ sweep (0.001, 0.01, 0.1, 1.0).
    Step 7  Generate all comparison plots and a final results summary.
"""

import os
import copy
import json
import torch
import numpy as np

from data       import load_all_clients, FlowDataset, FEATURE_COLUMNS, LABEL_COLUMN
from model      import ResidualMLP
from train      import local_train, evaluate, analyse_thresholds
from federated  import federated_train, fedprox_mu_sweep
from visualize  import (
    plot_training_curves,
    plot_algorithm_comparison,
    plot_mu_sweep,
    plot_confusion_matrices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATHS = {
    "ctu13": "data/ctu13_windowed.csv",
    "suee8": "data/suee8_windowed.csv",
    "unsw":  "data/unsw_windowed.csv",
    "caida": "data/caida_windowed.csv",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Federated learning hyperparameters
NUM_ROUNDS   = 20    # how many FL communication rounds
LOCAL_EPOCHS = 5     # max local epochs per client per round
LR           = 1e-3  # Adam learning rate
PATIENCE     = 5     # early stopping patience
INPUT_DIM    = 31    # must match number of features from Windower

# Output folder for plots and saved models
os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create a fresh untrained global model
# ─────────────────────────────────────────────────────────────────────────────

def fresh_model():
    """
    Always start FL experiments from the same random initialisation.
    We fix the seed so that differences between FedAvg and FedProx
    are due to the algorithm, not lucky/unlucky weight initialisation.
    """
    torch.manual_seed(42)
    return ResidualMLP(input_dim=INPUT_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: verify model architecture
# ─────────────────────────────────────────────────────────────────────────────

def verify_model():
    print("\n" + "="*55)
    print("  STEP 0 — Model architecture sanity check")
    print("="*55)
    model = fresh_model()
    dummy = torch.randn(16, INPUT_DIM)
    out   = model(dummy)
    assert out.shape == (16,),  f"Wrong output shape: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1]"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Output shape   : {out.shape}  ✓")
    print(f"  Output range   : [{out.min():.3f}, {out.max():.3f}]  ✓")
    print(f"  Total params   : {n_params:,}")
    print(f"  Device         : {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: load all clients
# ─────────────────────────────────────────────────────────────────────────────

def load_clients():
    print("\n" + "="*55)
    print("  STEP 1 — Loading client datasets")
    print("="*55)

    # Check all paths exist before loading
    missing = [p for p in DATASET_PATHS.values() if not os.path.exists(p)]
    if missing:
        print(f"\n  ✗ Missing CSV files: {missing}")
        print("  Please run your Windower first and place the output CSVs in data/")
        raise FileNotFoundError(f"Missing: {missing}")

    clients = load_all_clients(DATASET_PATHS, verbose=True)
    return clients


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: centralised baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_centralised_baseline(clients: dict):
    """
    Trains a single model on all clients' data pooled together.

    Why run this?
    --------------
    The centralised baseline represents the ceiling: the best performance
    we could achieve if we were willing to share all raw data.  Comparing
    the federated results to this baseline answers the core question:
    "how much does privacy-preserving FL cost us in detection performance?"

    If the federated model is close to the centralised baseline, it means
    we are getting near-optimal security performance without requiring any
    client to share its raw traffic.
    """
    from torch.utils.data import ConcatDataset, DataLoader

    print("\n" + "="*55)
    print("  STEP 2 — Centralised baseline (all data pooled)")
    print("="*55)
    print("  NOTE: This violates data privacy — it is only run as an")
    print("  upper-bound reference for comparing federated results.")

    # Pool all training sets into one combined DataLoader
    all_train = ConcatDataset([c["train"].dataset for c in clients.values()])
    all_test  = {name: c["test"] for name, c in clients.items()}

    # Compute an average pos_weight across clients (rough approximation)
    avg_pos_weight = np.mean([c["pos_weight"] for c in clients.values()])

    from torch.utils.data import random_split
    val_size = int(0.1 * len(all_train))
    train_size = len(all_train) - val_size
    train_sub, val_sub = random_split(
        all_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    central_train = DataLoader(train_sub, batch_size=256, shuffle=True)
    central_val   = DataLoader(val_sub,   batch_size=512, shuffle=False)

    model = fresh_model()
    trained_model, _ = local_train(
        model        = model,
        train_loader = central_train,
        val_loader   = central_val,
        pos_weight   = avg_pos_weight,
        device       = DEVICE,
        max_epochs   = 30,
        lr           = LR,
        patience     = 7,
        verbose      = True,
        client_name  = "centralised",
    )

    print("\n  Centralised baseline — per-client test evaluation:")
    central_metrics = {}
    for client_name, test_loader in all_test.items():
        m = evaluate(trained_model, test_loader, DEVICE,
                     client_name=f"centralised→{client_name}", verbose=True)
        central_metrics[client_name] = m

    return central_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step 3-5: FL experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_fl_experiments(clients: dict):
    print("\n" + "="*55)
    print("  STEPS 3-5 — Federated learning experiments")
    print("="*55)

    fl_results  = {}
    fl_histories = {}

    for algo, mu in [("fedavg", 0.0), ("fedprox", 0.01), ("fednova", 0.0)]:
        print(f"\n{'─'*55}")
        print(f"  Running {algo.upper()}  (μ={mu})")
        print(f"{'─'*55}")

        model = fresh_model()
        trained_model, history = federated_train(
            clients       = clients,
            global_model  = model,
            device        = DEVICE,
            num_rounds    = NUM_ROUNDS,
            local_epochs  = LOCAL_EPOCHS,
            lr            = LR,
            patience      = PATIENCE,
            proximal_mu   = mu,
            algorithm     = algo,
            eval_every    = 5,
            verbose       = True,
        )

        # Final evaluation on all test sets
        print(f"\n  Final evaluation — {algo.upper()}")
        algo_metrics = {}
        for client_name, client_data in clients.items():
            m = evaluate(
                trained_model, client_data["test"], DEVICE,
                client_name=f"{algo}→{client_name}", verbose=True
            )
            algo_metrics[client_name] = m

        fl_results[algo]   = algo_metrics
        fl_histories[algo] = history

        # Save the trained global model
        torch.save(
            trained_model.state_dict(),
            f"results/global_model_{algo}.pt"
        )
        print(f"  Model saved to results/global_model_{algo}.pt")

    return fl_results, fl_histories


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: FedProx μ sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_mu_sweep(clients: dict):
    print("\n" + "="*55)
    print("  STEP 6 — FedProx μ sweep")
    print("="*55)

    sweep_results = fedprox_mu_sweep(
        clients      = clients,
        device       = DEVICE,
        mu_values    = [0.001, 0.01, 0.1, 1.0],
        num_rounds   = 10,        # shorter run for the sweep
        local_epochs = LOCAL_EPOCHS,
        input_dim    = INPUT_DIM,
    )
    return sweep_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: plots and summary
# ─────────────────────────────────────────────────────────────────────────────

def generate_outputs(fl_results, fl_histories, sweep_results, central_metrics):
    print("\n" + "="*55)
    print("  STEP 7 — Generating plots and summary")
    print("="*55)

    # Training curves
    if "fedavg" in fl_histories and "fedprox" in fl_histories:
        plot_training_curves(
            fl_histories["fedavg"],
            fl_histories["fedprox"],
            save_path="results/training_curves.png"
        )

    # Algorithm comparison bar chart
    plot_algorithm_comparison(
        fl_results,
        metric="f1",
        save_path="results/algorithm_comparison_f1.png"
    )
    plot_algorithm_comparison(
        fl_results,
        metric="recall",
        save_path="results/algorithm_comparison_recall.png"
    )

    # μ sweep
    plot_mu_sweep(
        sweep_results,
        metric="f1",
        save_path="results/mu_sweep.png"
    )

    # Confusion matrices for the best algorithm
    # Pick the algorithm with the highest mean F1 across clients
    best_algo = max(
        fl_results.keys(),
        key=lambda a: np.mean([m["f1"] for m in fl_results[a].values()])
    )
    plot_confusion_matrices(
        fl_results[best_algo],
        save_path="results/confusion_matrices.png"
    )

    # ── Print final summary table ─────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL RESULTS SUMMARY")
    print("="*65)
    print(f"  {'':10}  {'':8}  {'F1':>8}  {'Recall':>8}  {'ROC-AUC':>9}  {'FNR':>7}")
    print(f"  {'─'*60}")

    # Centralised baseline
    for client, m in central_metrics.items():
        print(f"  {'centralised':10}  {client:8}  "
              f"{m['f1']:8.4f}  {m['recall']:8.4f}  "
              f"{m['roc_auc']:9.4f}  {m['false_neg_rate']:7.4f}")

    print(f"  {'─'*60}")

    # FL algorithms
    for algo in ["fedavg", "fedprox", "fednova"]:
        if algo not in fl_results:
            continue
        for client, m in fl_results[algo].items():
            print(f"  {algo:10}  {client:8}  "
                  f"{m['f1']:8.4f}  {m['recall']:8.4f}  "
                  f"{m['roc_auc']:9.4f}  {m['false_neg_rate']:7.4f}")
        print(f"  {'─'*60}")

    # Save results to JSON for later use
    def make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open("results/all_metrics.json", "w") as f:
        json.dump(make_serialisable({
            "centralised": central_metrics,
            "fl":          fl_results,
            "mu_sweep":    {str(k): v for k, v in sweep_results.items()},
        }), f, indent=2)
    print("\n  All metrics saved to results/all_metrics.json")

    print(f"\n  Best FL algorithm by mean F1: {best_algo.upper()}")
    mean_f1 = np.mean([m["f1"] for m in fl_results[best_algo].values()])
    print(f"  Mean F1 across all clients  : {mean_f1:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*55)
    print("  IDS FEDERATED LEARNING — FULL EXPERIMENT PIPELINE")
    print("█"*55)
    print(f"  Device : {DEVICE}")
    print(f"  Rounds : {NUM_ROUNDS}    Local epochs : {LOCAL_EPOCHS}")

    verify_model()
    clients         = load_clients()
    central_metrics = run_centralised_baseline(clients)
    fl_results, fl_histories = run_fl_experiments(clients)
    sweep_results   = run_mu_sweep(clients)
    generate_outputs(fl_results, fl_histories, sweep_results, central_metrics)

    print("\n  ✓ Experiment complete.  All outputs in results/")
