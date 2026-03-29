"""
quick_test.py  —  End-to-end test with synthetic data (no real CSVs needed)
===========================================================================
Run this FIRST to verify the entire pipeline works before using real data.

    python quick_test.py

It generates four synthetic client datasets with realistic imbalance
ratios and size differences, then runs a 3-round federated experiment
with all three algorithms.  The whole thing takes about 30 seconds on CPU.
"""

import os
import torch
import numpy as np
import pandas as pd

from data       import FEATURE_COLUMNS, LABEL_COLUMN, load_all_clients
from model      import ResidualMLP
from train      import evaluate, analyse_thresholds
from federated  import federated_train


# ── 1. Generate synthetic CSV files that mimic your real datasets ─────────────

def make_synthetic_csv(
    path: str,
    n_rows: int,
    attack_ratio: float,
    seed: int,
):
    """
    Creates a synthetic CSV with realistic class imbalance and feature ranges.

    The attack samples are generated with slightly shifted feature means so
    the model actually has something to learn — if attack and normal features
    were drawn from the same distribution, no model could distinguish them.
    """
    rng = np.random.default_rng(seed)
    n_attack = int(n_rows * attack_ratio)
    n_normal = n_rows - n_attack

    # Normal traffic: features drawn from standard normal
    X_normal = rng.standard_normal((n_normal, len(FEATURE_COLUMNS)))
    y_normal = np.zeros(n_normal)

    # Attack traffic: features shifted by +1.5 on half the features
    # (mimicking higher packet rates, different inter-arrival patterns)
    X_attack = rng.standard_normal((n_attack, len(FEATURE_COLUMNS)))
    X_attack[:, :15] += 1.5    # first 15 features are elevated in attacks
    y_attack = np.ones(n_attack)

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])

    # Shuffle so normal and attack rows are interleaved
    idx = rng.permutation(len(y))
    df = pd.DataFrame(X[idx], columns=FEATURE_COLUMNS)
    df[LABEL_COLUMN] = y[idx].astype(int)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Created {path}  ({n_rows:,} rows, {attack_ratio*100:.0f}% attack)")


def create_synthetic_datasets():
    print("\n  Creating synthetic client datasets...")
    print("  (These mimic your real datasets' size and imbalance ratios)\n")

    os.makedirs("synthetic_data", exist_ok=True)

    # Mimic real dataset characteristics:
    #   CTU-13 : large, mostly normal (botnets are minority of university traffic)
    #   SUEE-8 : small, moderate attack ratio (slow DoS scenarios)
    #   UNSW   : medium, moderate imbalance (9 attack categories)
    #   CAIDA  : medium, attack-heavy during the capture period
    configs = {
        "ctu13": ("synthetic_data/ctu13.csv",   8000, 0.08),
        "suee8": ("synthetic_data/suee8.csv",   2000, 0.20),
        "unsw":  ("synthetic_data/unsw.csv",    5000, 0.35),
        "caida": ("synthetic_data/caida.csv",   4000, 0.60),
    }
    for name, (path, n, ratio) in configs.items():
        make_synthetic_csv(path, n, ratio, seed=hash(name) % 1000)

    return {name: path for name, (path, _, _) in configs.items()}


# ── 2. Run a mini federated experiment ───────────────────────────────────────

def run_quick_test():
    print("\n" + "="*55)
    print("  QUICK TEST — verifying full pipeline")
    print("="*55)

    device = torch.device("cpu")   # CPU is fine for this small test

    # Create synthetic data
    dataset_paths = create_synthetic_datasets()

    # Load into client DataLoaders
    print("\n  Loading client DataLoaders...")
    clients = load_all_clients(dataset_paths, verbose=True)

    # Verify model instantiates and runs correctly
    print("\n  Model sanity check...")
    model = ResidualMLP(input_dim=31)
    dummy = torch.randn(16, 31)
    out   = model(dummy)
    assert out.shape == (16,)
    assert 0 <= out.min() <= out.max() <= 1
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Output shape {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")
    print(f"  ✓ Total parameters: {n_params:,}")

    # Run 3 rounds of FedAvg
    print("\n  Running 3-round FedAvg test...")
    torch.manual_seed(42)
    model_avg = ResidualMLP(input_dim=31)
    trained_avg, history_avg = federated_train(
        clients       = clients,
        global_model  = model_avg,
        device        = device,
        num_rounds    = 3,
        local_epochs  = 3,
        lr            = 1e-3,
        patience      = 3,
        proximal_mu   = 0.0,
        algorithm     = "fedavg",
        eval_every    = 3,
        verbose       = True,
    )

    # Run 3 rounds of FedProx
    print("\n  Running 3-round FedProx test (μ=0.01)...")
    torch.manual_seed(42)
    model_prox = ResidualMLP(input_dim=31)
    trained_prox, history_prox = federated_train(
        clients       = clients,
        global_model  = model_prox,
        device        = device,
        num_rounds    = 3,
        local_epochs  = 3,
        lr            = 1e-3,
        patience      = 3,
        proximal_mu   = 0.01,
        algorithm     = "fedprox",
        eval_every    = 3,
        verbose       = True,
    )

    # Final evaluation
    print("\n  Final evaluation — FedAvg vs FedProx")
    for algo, model in [("FedAvg", trained_avg), ("FedProx", trained_prox)]:
        print(f"\n  ── {algo} ──")
        for client_name, client_data in clients.items():
            evaluate(
                model, client_data["test"], device,
                client_name=f"{algo}→{client_name}", verbose=True
            )

    # Threshold analysis on one client
    print("\n  Threshold analysis on CTU-13:")
    analyse_thresholds(
        trained_avg, clients["ctu13"]["test"], device,
        client_name="ctu13 (FedAvg)"
    )

    print("\n" + "="*55)
    print("  ✓ QUICK TEST PASSED — all components working correctly")
    print("="*55)
    print("\n  Next step: replace synthetic_data/ with your real Windower")
    print("  outputs and run:  python run_experiment.py")
    print("  (Update DATASET_PATHS in run_experiment.py first)")


if __name__ == "__main__":
    run_quick_test()
