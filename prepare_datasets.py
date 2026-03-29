"""
prepare_datasets.py  —  Merge, label, and analyse the Windower datasets
========================================================================
Run this script once before any training. It reads the three-file format
that Patrick provided (train.csv, test.csv, test-attack-ips.txt) for each
dataset, merges and labels everything correctly, and saves one clean CSV
per dataset that your existing load_client_data() function can read directly.

Usage
-----
    python prepare_datasets.py

Output
------
    data/ctu13_labeled.csv
    data/suee8_labeled.csv
    data/unsw_labeled.csv

Then update DATASET_PATHS in run_experiment.py to point at these files.
"""

import os
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — update these paths to match where you put the files
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "ctu13": {
        "train":      "raw/ctu13/train.csv",
        "test":       "raw/ctu13/test.csv",
        "attack_ips": "raw/ctu13/test-attack-ips.txt",
        "output":     "data/ctu13_labeled.csv",
    },
    "suee8": {
        "train":      "raw/suee8/train.csv",
        "test":       "raw/suee8/test.csv",
        "attack_ips": "raw/suee8/test-attack-ips.txt",
        "output":     "data/suee8_labeled.csv",
    },
    "unsw": {
        "train":      "raw/unsw/train.csv",
        "test":       "raw/unsw/test.csv",
        "attack_ips": "raw/unsw/test-attack-ips.txt",
        "output":     "data/unsw_labeled.csv",
    },
}

# Columns the Windower paper explicitly drops before classification (Sec. IV-A).
# - src_ip     : dropped to prevent the model from memorising attacker IPs
#                instead of learning traffic patterns.  If src_ip stayed in,
#                the model would just say "this IP was an attacker last time"
#                rather than learning anything about the 31 statistics.
# - window_count: metadata about how many windows were summarised.  Useful
#                for postprocessing confidence but not for classification.
# - window_span : the difference between first and last window ID.  Same reason.
COLS_TO_DROP = ["src_ip", "window_count", "window_span"]

# The column name your model expects for the binary label
LABEL_COLUMN = "label"

# ─────────────────────────────────────────────────────────────────────────────
# Core preparation function
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(name: str, paths: dict) -> pd.DataFrame:
    """
    Merges and labels one dataset from the three-file Windower format.

    The logic in plain language:
    ----------------------------
    1. Read train.csv — every row here is a benign window by construction
       (Patrick separated benign-only for KitNet training), so label = 0.

    2. Read test.csv — rows here can be either benign or attack windows.
       We do NOT know from the CSV itself which is which.  The knowledge
       lives in test-attack-ips.txt.

    3. Read test-attack-ips.txt into a Python set.  A set gives us O(1)
       lookup: checking "is this IP in the set?" takes the same time
       whether the set has 5 IPs or 5 million.

    4. For each row in test.csv, check whether its src_ip appears in the
       attack IPs set.  If yes → label = 1 (attack).  If no → label = 0.

    5. Concatenate train (all label=0) and labeled test into one DataFrame.

    6. Drop the three metadata columns the paper excludes.

    7. Sanity-check that we actually have both classes present.

    Returns the merged, labeled DataFrame (also saves it to disk).
    """

    print(f"\n{'═' * 55}")
    print(f"  Processing: {name.upper()}")
    print(f"{'═' * 55}")

    # ── Step 1: load train (benign only) ─────────────────────────────────────
    print(f"  Loading train.csv ...")
    train_df = pd.read_csv(paths["train"])
    train_df[LABEL_COLUMN] = 0   # every row is benign by design

    print(f"    Rows in train.csv  : {len(train_df):>8,}  (all benign)")

    # ── Step 2: load test (unlabeled mix) ────────────────────────────────────
    print(f"  Loading test.csv ...")
    test_df = pd.read_csv(paths["test"])

    print(f"    Rows in test.csv   : {len(test_df):>8,}  (unlabeled)")

    # Quick check: verify src_ip exists in both files before we go further.
    # If Patrick's CSV uses a different column name for the source IP,
    # you will get a clear error message here rather than silent wrong labels.
    for df_name, df in [("train.csv", train_df), ("test.csv", test_df)]:
        if "src_ip" not in df.columns:
            raise ValueError(
                f"Column 'src_ip' not found in {df_name}.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Update COLS_TO_DROP and the isin() call to use the "
                f"correct IP column name."
            )

    # ── Step 3: load attacker IPs ────────────────────────────────────────────
    with open(paths["attack_ips"], "r") as f:
        # Strip whitespace and skip empty lines.
        # Using a set (not a list) is important for speed.
        attack_ips = set(line.strip() for line in f if line.strip())

    print(f"    Known attacker IPs : {len(attack_ips):>8,}")

    # ── Step 4: label test.csv rows ──────────────────────────────────────────
    # pandas .isin() checks each value in the src_ip column against the set.
    # The result is a boolean Series (True/False), which .astype(int) converts
    # to 1/0.  This is vectorised — much faster than a Python for-loop.
    test_df[LABEL_COLUMN] = test_df["src_ip"].isin(attack_ips).astype(int)

    n_attack_in_test = int(test_df[LABEL_COLUMN].sum())
    n_benign_in_test = len(test_df) - n_attack_in_test
    print(f"    After labeling test.csv:")
    print(f"      Attack windows : {n_attack_in_test:>8,}")
    print(f"      Benign windows : {n_benign_in_test:>8,}")

    # Warn if no attack rows were found — this would mean the IP format
    # in the CSV does not match the format in the .txt file (e.g. one
    # has "192.168.1.1" and the other has "192.168.001.001")
    if n_attack_in_test == 0:
        print(f"\n  ⚠️  WARNING: No attack rows found in test.csv after labeling!")
        print(f"     First 5 IPs from test.csv    : "
              f"{test_df['src_ip'].unique()[:5].tolist()}")
        print(f"     First 5 IPs from attack file : "
              f"{sorted(attack_ips)[:5]}")
        print(f"     These formats must match exactly (spaces, leading zeros, etc.)")

    # ── Step 5: merge ─────────────────────────────────────────────────────────
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # ── Step 6: drop metadata columns ────────────────────────────────────────
    # errors="ignore" means if a column doesn't exist (e.g. the CSV already
    # had it removed), pandas won't crash — it just skips it silently.
    cols_present = [c for c in COLS_TO_DROP if c in combined.columns]
    combined = combined.drop(columns=cols_present, errors="ignore")

    if cols_present:
        print(f"\n  Dropped metadata columns: {cols_present}")

    # ── Step 7: final sanity checks ───────────────────────────────────────────
    n_total   = len(combined)
    n_attack  = int(combined[LABEL_COLUMN].sum())
    n_benign  = n_total - n_attack
    pct_atk   = 100 * n_attack / max(n_total, 1)
    pos_weight = n_benign / max(n_attack, 1)

    print(f"\n  ── Final merged dataset ──────────────────────────────")
    print(f"  Total windows  : {n_total:>8,}")
    print(f"  Benign (0)     : {n_benign:>8,}  ({100 - pct_atk:.1f}%)")
    print(f"  Attack (1)     : {n_attack:>8,}  ({pct_atk:.1f}%)")
    print(f"  pos_weight     : {pos_weight:>8.2f}  "
          f"(attack class will be weighted {pos_weight:.1f}x in the loss)")
    print(f"  Feature columns: {len(combined.columns) - 1}")

    # Check for NaN or infinite values that could silently break training
    nan_count = combined.drop(columns=[LABEL_COLUMN]).isnull().sum().sum()
    inf_count = np.isinf(
        combined.drop(columns=[LABEL_COLUMN]).select_dtypes(include=np.number)
    ).sum().sum()

    if nan_count > 0:
        print(f"\n  ⚠️  Found {nan_count:,} NaN values — "
              f"load_client_data() will drop these rows automatically.")
    if inf_count > 0:
        print(f"\n  ⚠️  Found {inf_count:,} infinite values — "
              f"load_client_data() will drop these rows automatically.")

    # Check that label column is exactly 0 and 1 with no surprises
    unique_labels = sorted(combined[LABEL_COLUMN].unique().tolist())
    if unique_labels not in [[0, 1], [0], [1]]:
        print(f"\n  ⚠️  Unexpected label values: {unique_labels}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(paths["output"]), exist_ok=True)
    combined.to_csv(paths["output"], index=False)
    print(f"\n  ✓  Saved to: {paths['output']}")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_federation_summary(all_stats: list):
    """
    Prints a clean summary table comparing all clients side by side.

    This table is directly useful for your project report — it is the
    dataset characterisation that justifies why your federation is non-IID
    and why the class imbalance varies across clients.
    """
    print(f"\n\n{'═' * 65}")
    print(f"  FEDERATION SUMMARY — all clients")
    print(f"{'═' * 65}")
    print(f"  {'Client':<10} {'Total':>10} {'Benign':>10} "
          f"{'Attack':>10} {'Atk%':>7} {'pos_weight':>12}")
    print(f"  {'─' * 60}")

    total_all = total_benign = total_attack = 0

    for s in all_stats:
        print(
            f"  {s['name']:<10} {s['total']:>10,} {s['benign']:>10,} "
            f"{s['attack']:>10,} {s['pct_atk']:>6.1f}% {s['pos_weight']:>12.2f}"
        )
        total_all    += s["total"]
        total_benign += s["benign"]
        total_attack += s["attack"]

    print(f"  {'─' * 60}")
    pct = 100 * total_attack / max(total_all, 1)
    pw  = total_benign / max(total_attack, 1)
    print(f"  {'TOTAL':<10} {total_all:>10,} {total_benign:>10,} "
          f"{total_attack:>10,} {pct:>6.1f}% {pw:>12.2f}")
    print(f"{'═' * 65}")

    print(f"""
  How to read this table for your report:
  ─────────────────────────────────────────────────────────────
  Total        : number of windowed feature vectors per client
  Benign (0)   : windows from confirmed legitimate source IPs
  Attack (1)   : windows from confirmed attacker source IPs
  Atk%         : percentage of attack windows in the full dataset
  pos_weight   : the value passed to the weighted loss function.
                 A pos_weight of 12.5 means the model is penalised
                 12.5x more for missing a real attack than for a
                 false alarm.  This corrects class imbalance without
                 modifying the data itself.

  The variation in Atk% across clients is not a problem —
  it is the non-IID property that makes this a meaningful
  federated learning experiment.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Dataset preparation — Windower three-file format → supervised CSV")
    print("=" * 65)

    all_stats = []

    for name, paths in DATASETS.items():

        # Check all required files exist before starting
        missing = [k for k, p in paths.items()
                   if k != "output" and not os.path.exists(p)]
        if missing:
            print(f"\n  ✗ Skipping {name.upper()} — missing files:")
            for m in missing:
                print(f"    {paths[m]}")
            print(f"    Update the paths in DATASETS at the top of this script.")
            continue

        df = prepare_dataset(name, paths)

        n_total  = len(df)
        n_attack = int(df[LABEL_COLUMN].sum())
        n_benign = n_total - n_attack

        all_stats.append({
            "name":       name,
            "total":      n_total,
            "benign":     n_benign,
            "attack":     n_attack,
            "pct_atk":    100 * n_attack / max(n_total, 1),
            "pos_weight": n_benign / max(n_attack, 1),
        })

    if all_stats:
        print_federation_summary(all_stats)

    print("\n  Next step: run  python run_experiment.py")
    print("  (Make sure DATASET_PATHS in run_experiment.py points to data/*.csv)\n")
