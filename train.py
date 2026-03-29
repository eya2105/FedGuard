"""
train.py  —  Local training loop, evaluation, and early stopping
================================================================
This module handles everything that happens on one client during its
local training phase in a federated round.

The key insight to keep in mind: this code runs independently on each
of the four clients.  No client sees another client's data.  The only
thing that travels between clients and the server is the model's weight
tensors, not the training data itself.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Early stopping helper
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors validation loss and stops training when it stops improving.

    Why do we need this in a federated setting?
    -------------------------------------------
    Each client runs N local epochs before sending its weights to the
    server.  If we allow too many epochs, a client can overfit to its
    local data distribution.  When that overfitted model is then averaged
    with the other three clients, the global model gets pulled toward
    patterns that are specific to one network environment rather than
    general across all four.

    Early stopping prevents this by halting local training the moment the
    client's model starts memorising its local data instead of learning
    generalisable patterns.

    The patience parameter controls how tolerant we are: patience=5 means
    "stop if validation loss has not improved in the last 5 epochs."
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta    # minimum improvement to count as "better"
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_weights = None       # snapshot of the best model so far

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves a deep copy of model weights whenever validation loss improves.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss    = val_loss
            self.counter      = 0
            # Deep copy: we save the model state, not a reference to it,
            # so subsequent training epochs don't overwrite the saved snapshot
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Loads the best-seen weights back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  One training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    device:       torch.device,
    proximal_mu:  float = 0.0,
    global_params: Optional[list] = None,
) -> float:
    """
    Runs one full pass through the training data.

    proximal_mu and global_params together implement the FedProx term.
    When proximal_mu=0 this function behaves exactly like standard FedAvg.

    The FedProx proximal term explained:
    -------------------------------------
    In standard SGD (and FedAvg), each local training step moves the model
    purely in the direction that reduces the local loss.  The problem is
    that with non-IID data, each client's local loss landscape points in a
    different direction — CTU-13's loss surface "wants" the model to look
    like a botnet detector, SUEE's "wants" it to look like a slow DoS
    detector, and so on.  Over several epochs, the clients drift apart.

    FedProx adds a second term to the loss:
        total_loss = local_loss + (μ/2) × ||w − w_global||²

    The second term is the squared distance between the current local
    weights (w) and the global weights sent at the start of the round
    (w_global).  It penalises the local model for moving far from the
    global model.  As μ increases, the "leash" gets tighter and client
    drift is suppressed more aggressively.

    Returns
    -------
    Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)

        # Standard binary cross-entropy loss
        loss = criterion(predictions, y_batch)

        # ── FedProx proximal term ─────────────────────────────────────────
        # Only active when proximal_mu > 0 and global parameters are provided
        if proximal_mu > 0 and global_params is not None:
            prox_term = 0.0
            for local_param, global_param in zip(
                model.parameters(), global_params
            ):
                # ||w_local − w_global||² for each parameter tensor
                prox_term += torch.norm(local_param - global_param.detach()) ** 2
            loss = loss + (proximal_mu / 2.0) * prox_term

        loss.backward()

        # Gradient clipping: prevents any single update step from being
        # explosively large, which can happen early in training or when
        # a client has a very unusual batch.  max_norm=1.0 is a conservative
        # standard value.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Validation loss calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_val_loss(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """
    Computes average loss on the validation set without updating weights.
    torch.no_grad() disables gradient computation — we don't need it for
    evaluation and it saves memory and time.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            total_loss += criterion(predictions, y_batch).item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full local training (called once per FL round per client)
# ─────────────────────────────────────────────────────────────────────────────

def local_train(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    pos_weight:    float,
    device:        torch.device,
    max_epochs:    int   = 10,
    lr:            float = 1e-3,
    patience:      int   = 5,
    proximal_mu:   float = 0.0,
    verbose:       bool  = True,
    client_name:   str   = "client",
) -> Tuple[nn.Module, Dict]:
    """
    Trains a model locally for up to max_epochs epochs, with early stopping.
    This is the function called on each client during every FL round.

    Parameters
    ----------
    model        : the model (already initialised with the current global weights)
    train_loader : this client's training DataLoader
    val_loader   : this client's validation DataLoader
    pos_weight   : the normal/attack ratio for this client's training data
                   (computed in data.py, unique per client)
    device       : cpu or cuda
    max_epochs   : upper bound on local epochs per FL round.  In practice
                   early stopping usually triggers before this.
    lr           : learning rate.  1e-3 is the Adam default and works well
                   here.  We don't need to tune this aggressively because
                   each client only trains for a few local epochs.
    patience     : epochs without validation improvement before stopping
    proximal_mu  : FedProx regularisation strength.  0.0 = FedAvg behaviour.
    verbose      : whether to print per-epoch metrics

    Returns
    -------
    model  : trained model (with best weights restored)
    history: dict with train_losses and val_losses lists for plotting
    """

    model = model.to(device)

    # Weighted BCE loss.
    # BCELoss expects predictions already passed through sigmoid.
    # We pass pos_weight as a tensor: the attack class is weighted
    # pos_weight times more heavily than the normal class.
    pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(weight=None)   # standard BCE first...

    # ...but we scale the loss for attack samples by constructing a
    # per-sample weight tensor during each batch.  This is equivalent
    # to BCEWithLogitsLoss(pos_weight=...) but works with our sigmoid output.
    # Simpler approach: just use the pos_weight in a custom way below.

    # Actually, let's use the cleanest approach: BCEWithLogitsLoss with
    # sigmoid moved to loss rather than model output.  We need to adjust
    # the model's forward to not apply sigmoid for this, OR use BCELoss
    # with manual weighting.  Here we do manual weighting for clarity:

    def weighted_bce(preds, targets):
        # preds are already in [0,1] from the sigmoid in model.forward()
        eps = 1e-7   # prevent log(0)
        loss_pos = -targets      * torch.log(preds + eps)        # attack samples
        loss_neg = -(1-targets)  * torch.log(1 - preds + eps)    # normal samples
        # Attack samples get multiplied by pos_weight
        weights = targets * pos_weight + (1 - targets) * 1.0
        return (weights * (loss_pos + loss_neg)).mean()

    # Adam optimiser: adaptive learning rates per parameter, robust default
    # for neural networks on tabular data.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler: reduces LR by 50% if val loss doesn't improve
    # for 3 epochs.  This prevents the model from oscillating around a
    # minimum without converging.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    early_stopper = EarlyStopping(patience=patience)

    # Snapshot of global weights for the FedProx proximal term.
    # These stay fixed throughout local training — they are the "anchor."
    global_params = [p.clone().detach() for p in model.parameters()] \
                    if proximal_mu > 0 else None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, weighted_bce,
            device, proximal_mu, global_params
        )
        val_loss = compute_val_loss(model, val_loader, weighted_bce, device)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(
                f"  [{client_name}]  epoch {epoch:02d}/{max_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

        # Check for early stopping
        if early_stopper.step(val_loss, model):
            if verbose:
                print(f"  [{client_name}]  early stop at epoch {epoch}")
            break

    # Restore the weights from the best-seen validation checkpoint
    early_stopper.restore_best(model)

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation on the test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:        nn.Module,
    test_loader:  DataLoader,
    device:       torch.device,
    threshold:    float = 0.5,
    client_name:  str   = "client",
    verbose:      bool  = True,
) -> Dict:
    """
    Evaluates the model on a test set and returns a comprehensive metrics dict.

    Why these specific metrics and not just accuracy?
    -------------------------------------------------
    With imbalanced datasets, a model that always predicts "normal" would
    achieve very high accuracy (e.g. 95% if 5% of traffic is attacks) while
    being completely useless as a security tool.  We need metrics that
    specifically measure performance on the minority (attack) class:

    Precision: of all windows we called "attack," how many were real attacks?
               Low precision = many false alarms.
    Recall:    of all real attacks, how many did we catch?
               Low recall = many missed attacks (false negatives).
               In security, THIS is the critical metric — missing a real
               attack is usually more costly than a false alarm.
    F1:        harmonic mean of precision and recall.  Good overall measure
               when both false alarms and missed attacks matter.
    ROC-AUC:   measures how well the model separates attacks from normal
               traffic across ALL possible thresholds, not just the one we
               chose.  1.0 = perfect, 0.5 = random.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch.to(device)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)
    all_preds  = (all_probs >= threshold).astype(int)

    # Confusion matrix gives us TP, FP, TN, FN
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    metrics = {
        "precision":   precision_score(all_labels, all_preds, zero_division=0),
        "recall":      recall_score(all_labels, all_preds, zero_division=0),
        "f1":          f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc":     roc_auc_score(all_labels, all_probs),
        "accuracy":    (tp + tn) / len(all_labels),
        "false_neg_rate": fn / max(fn + tp, 1),   # fraction of attacks missed
        "false_pos_rate": fp / max(fp + tn, 1),   # fraction of normal flagged
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    if verbose:
        print(f"\n  ┌─ Evaluation: {client_name} {'─'*30}")
        print(f"  │  F1          : {metrics['f1']:.4f}")
        print(f"  │  Precision   : {metrics['precision']:.4f}")
        print(f"  │  Recall      : {metrics['recall']:.4f}")
        print(f"  │  ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"  │  Accuracy    : {metrics['accuracy']:.4f}")
        print(f"  │  False neg   : {metrics['false_neg_rate']:.4f}  "
              f"({fn} missed attacks)")
        print(f"  │  False pos   : {metrics['false_pos_rate']:.4f}  "
              f"({fp} false alarms)")
        print(f"  └{'─'*45}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Human-in-the-loop threshold analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_thresholds(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      torch.device,
    client_name: str = "client",
):
    """
    Sweeps the confidence threshold from 0.1 to 0.9 and prints how
    precision, recall, and human review rate change.

    This is important for setting the three zones:
      - score < low_thresh  → auto-normal
      - score > high_thresh → auto-alert
      - in between          → human review

    In deployment you want to tune these thresholds based on the cost
    ratio of a missed attack versus a false alarm in your specific network.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch.to(device)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)

    print(f"\n  Threshold analysis — {client_name}")
    print(f"  {'threshold':>10} {'precision':>10} {'recall':>10} "
          f"{'f1':>8} {'review%':>10}")
    print(f"  {'─'*55}")

    for thresh in np.arange(0.1, 1.0, 0.1):
        preds     = (all_probs >= thresh).astype(int)
        prec      = precision_score(all_labels, preds, zero_division=0)
        rec       = recall_score(all_labels, preds, zero_division=0)
        f1        = f1_score(all_labels, preds, zero_division=0)
        uncertain = ((all_probs >= 0.3) & (all_probs < thresh)).mean()
        print(
            f"  {thresh:>10.1f} {prec:>10.4f} {rec:>10.4f} "
            f"{f1:>8.4f} {uncertain*100:>9.1f}%"
        )
