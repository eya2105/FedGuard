"""
data.py  —  Data loading, preprocessing and client dataset preparation
=======================================================================
Handles everything between raw CSV files and PyTorch DataLoaders.

Your Windower outputs CSV files where each row is one time window of one
4-tuple flow, with 31 statistical feature columns and one label column
(0 = normal, 1 = attack).  This module reads those CSVs, preprocesses
them, and packages each dataset as a separate federated client.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Column configuration
# ─────────────────────────────────────────────────────────────────────────────

# These are the 31 feature names the Windower produces (4-tuple aggregation).
# Adjust this list to exactly match your CSV column names.
# The order matters because we later scale by column position.
FEATURE_COLUMNS = [
    # Packet-level statistics
    "pkt_count",           "byte_count",
    "mean_pkt_len",        "std_pkt_len",
    "min_pkt_len",         "max_pkt_len",
    # Inter-arrival time statistics
    "mean_iat",            "std_iat",
    "min_iat",             "max_iat",
    # Forward/backward direction split
    "fwd_pkt_count",       "bwd_pkt_count",
    "fwd_byte_count",      "bwd_byte_count",
    "fwd_mean_pkt_len",    "bwd_mean_pkt_len",
    "fwd_mean_iat",        "bwd_mean_iat",
    # TCP flag counts (0 for non-TCP flows)
    "fin_flag_cnt",        "syn_flag_cnt",
    "rst_flag_cnt",        "psh_flag_cnt",
    "ack_flag_cnt",        "urg_flag_cnt",
    # Flow-level summaries
    "flow_duration",       "active_mean",
    "idle_mean",           "pkt_rate",
    "byte_rate",           "down_up_ratio",
    # Window metadata
    "window_size_sec",
]

LABEL_COLUMN = "label"   # 0 = normal, 1 = attack


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PyTorch Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FlowDataset(Dataset):
    """
    Wraps numpy arrays (features, labels) into a PyTorch Dataset.
    Nothing fancy — just converts to tensors on access so that the
    DataLoader can batch them automatically.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Single-client loading function
# ─────────────────────────────────────────────────────────────────────────────

def load_client_data(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, StandardScaler]:
    """
    Reads one client's CSV, splits into train/val/test, fits a scaler on
    the training split (and applies it to val and test), then wraps
    everything in DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, pos_weight, scaler

    pos_weight (float):
        The ratio normal / attack in the training split.  Pass this to
        BCEWithLogitsLoss as pos_weight so the model treats each attack
        sample as proportionally more important.  This is the correct
        way to handle class imbalance without touching the data itself.

    scaler (StandardScaler):
        Fitted on training data only.  Save this if you later need to
        preprocess new live traffic through the same normalisation.

    Why fit the scaler on training data only?
    -----------------------------------------
    If you fit the scaler on the full dataset (including test), you leak
    information about the test distribution into the model's preprocessing
    — the model effectively "sees" test data before being evaluated on it.
    This inflates performance metrics and gives you a falsely optimistic
    picture of how the model generalises.
    """

    df = pd.read_csv(csv_path)

    # ── basic validation ──────────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV at '{csv_path}' is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows with NaN or infinite values — these come from division-by-zero
    # in some Windower statistics (e.g. std when only one packet in window)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.float32)

    # ── stratified splits ─────────────────────────────────────────────────────
    # Stratified = each split preserves the original attack/normal ratio.
    # This is important with imbalanced data: a random split might put all
    # attacks in training and none in test, making evaluation meaningless.

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # val_size is relative to the remaining (train+val) portion
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac, stratify=y_train_val, random_state=random_state
    )

    # ── feature scaling ───────────────────────────────────────────────────────
    # StandardScaler maps each feature to zero mean and unit variance.
    # Why do we need this?  Our 31 features span very different numerical
    # ranges: pkt_count might be in the hundreds, flow_duration in the
    # millions of microseconds, flag counts in single digits.  Without
    # scaling, the loss gradient is dominated by whichever feature has the
    # largest numerical values, regardless of whether that feature is actually
    # more informative.  Scaling puts all features on equal footing.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit AND transform on train
    X_val   = scaler.transform(X_val)          # transform only
    X_test  = scaler.transform(X_test)         # transform only

    # ── class weight for the loss function ───────────────────────────────────
    n_normal = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()

    if n_attack == 0:
        raise ValueError(
            f"No attack samples found in training split of '{csv_path}'.\n"
            "Either this dataset has no attack labels or the column is mislabeled."
        )

    # pos_weight = how many times more important an attack sample is than a
    # normal sample.  If 95% normal → pos_weight = 19.
    pos_weight = float(n_normal) / float(n_attack)

    if verbose:
        dataset_name = os.path.basename(csv_path)
        print(f"\n{'─'*55}")
        print(f"  Dataset  : {dataset_name}")
        print(f"  Total    : {len(y):,}  samples")
        print(f"  Train    : {len(y_train):,}  "
              f"(normal={int(n_normal):,}, attack={int(n_attack):,})")
        print(f"  Val      : {len(y_val):,}")
        print(f"  Test     : {len(y_test):,}")
        print(f"  pos_weight (attack class) : {pos_weight:.2f}")
        print(f"{'─'*55}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # batch_size=256 is a good default for tabular data at this scale.
    # shuffle=True on training so the model doesn't learn the order of rows.
    # pin_memory=True speeds up CPU→GPU transfer if you're using a GPU.
    train_loader = DataLoader(
        FlowDataset(X_train, y_train),
        batch_size=256, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        FlowDataset(X_val, y_val),
        batch_size=512, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        FlowDataset(X_test, y_test),
        batch_size=512, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader, pos_weight, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Multi-client loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_clients(
    dataset_paths: Dict[str, str],
    **kwargs,
) -> Dict[str, dict]:
    """
    Loads all four federated clients in one call.

    Parameters
    ----------
    dataset_paths : dict
        Maps a client name to its CSV file path, e.g.
        {
            "ctu13"   : "data/ctu13_windowed.csv",
            "suee8"   : "data/suee8_windowed.csv",
            "unsw"    : "data/unsw_windowed.csv",
            "caida"   : "data/caida_windowed.csv",
        }

    Returns
    -------
    clients : dict
        {
          "ctu13": {
            "train": DataLoader,
            "val":   DataLoader,
            "test":  DataLoader,
            "pos_weight": float,
            "scaler": StandardScaler,
          },
          ...
        }
    """
    clients = {}
    for name, path in dataset_paths.items():
        train_loader, val_loader, test_loader, pos_weight, scaler = \
            load_client_data(path, **kwargs)
        clients[name] = {
            "train":      train_loader,
            "val":        val_loader,
            "test":       test_loader,
            "pos_weight": pos_weight,
            "scaler":     scaler,
        }
    return clients
def prepare_windower_dataset(
    train_csv: str,
    test_csv: str,
    attack_ips_txt: str,
    output_csv: str,
    verbose: bool = True,
):
    """
    Converts the Windower three-file format into a single labeled CSV
    suitable for supervised learning.

    The key transformation: Patrick's train.csv is benign-only (designed
    for anomaly detection). We label it all as 0, then label test.csv
    using the attack IPs file, merge everything, and save a single CSV
    where both classes are present. Your existing load_client_data()
    function then handles the stratified train/val/test re-split.

    Parameters
    ----------
    train_csv       : path to the Windower train.csv (benign only)
    test_csv        : path to the Windower test.csv (benign + attack, unlabeled)
    attack_ips_txt  : path to test-attack-ips.txt (one IP per line)
    output_csv      : where to save the merged, labeled CSV
    """
    import pandas as pd

    # Step 1: read the known attacker IPs into a set.
    # Using a set gives O(1) lookup — much faster than checking
    # a list when you have millions of rows.
    with open(attack_ips_txt, "r") as f:
        attack_ips = set(line.strip() for line in f if line.strip())

    if verbose:
        print(f"  Known attacker IPs loaded: {len(attack_ips)}")

    # Step 2: label the training portion — all benign by design.
    train_df = pd.read_csv(train_csv)
    train_df[LABEL_COLUMN] = 0

    # Step 3: label the test portion by IP lookup.
    # Any window whose source IP is a known attacker gets label=1.
    # Everything else is label=0 (benign).
    test_df = pd.read_csv(test_csv)
    test_df[LABEL_COLUMN] = test_df["src_ip"].isin(attack_ips).astype(int)

    # Step 4: merge into one pool.
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Step 5: drop the columns the Windower paper excludes before
    # classification (Section IV-A). src_ip is dropped to prevent
    # evaluation bias — the model should classify based on traffic
    # statistics, not by memorising which IPs were attackers in the
    # training data. window_count and window_span are metadata useful
    # for postprocessing but not informative for the classifier itself.
    cols_to_drop = ["src_ip", "window_count", "window_span"]
    combined = combined.drop(
        columns=[c for c in cols_to_drop if c in combined.columns],
        errors="ignore"
    )

    combined.to_csv(output_csv, index=False)

    if verbose:
        n_attack = int(combined[LABEL_COLUMN].sum())
        n_normal = len(combined) - n_attack
        ratio = n_normal / max(n_attack, 1)
        print(f"  Total rows  : {len(combined):,}")
        print(f"  Normal rows : {n_normal:,}")
        print(f"  Attack rows : {n_attack:,}")
        print(f"  pos_weight will be approximately: {ratio:.2f}")
        print(f"  Saved to: {output_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    # Generate a tiny synthetic CSV to test the pipeline end-to-end
    # without needing real data files
    n_rows = 1000
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.standard_normal((n_rows, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS
    )
    # Simulate class imbalance: 90% normal, 10% attack
    data[LABEL_COLUMN] = rng.choice([0, 1], size=n_rows, p=[0.90, 0.10])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        data.to_csv(f, index=False)
        tmp_path = f.name

    train_loader, val_loader, test_loader, pos_weight, scaler = \
        load_client_data(tmp_path)

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nFeature batch shape : {X_batch.shape}")
    print(f"Label batch shape   : {y_batch.shape}")
    print(f"pos_weight          : {pos_weight:.2f}")
    os.unlink(tmp_path)
