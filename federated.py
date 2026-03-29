"""
federated.py  —  Federated aggregation: FedAvg, FedProx, FedNova
=================================================================
This module lives on the server side of the federation.  It receives
weight updates from all clients after their local training phase,
aggregates them into a new global model, and sends it back.

The three algorithms implemented here correspond directly to the
discussion in the project documentation:
  - FedAvg    : baseline, sample-proportional weighted average
  - FedProx   : same aggregation but local training uses proximal term
                (the proximal term is in train.py; here aggregation is
                identical to FedAvg — the difference is in how clients
                trained, not in how the server aggregates)
  - FedNova   : normalises each client's update by the number of local
                gradient steps before averaging
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from model import ResidualMLP
from train import local_train, evaluate


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Weight extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_weights(model: nn.Module) -> List[torch.Tensor]:
    """Returns a list of detached CPU tensor copies of all model parameters."""
    return [p.detach().cpu().clone() for p in model.parameters()]


def set_weights(model: nn.Module, weights: List[torch.Tensor]):
    """Loads a list of tensors back into a model's parameters in-place."""
    with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.copy_(weight)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────────────

def fedavg_aggregate(
    client_weights: List[List[torch.Tensor]],
    client_sizes:   List[int],
) -> List[torch.Tensor]:
    """
    Computes the sample-proportional weighted average of client model weights.

    Why sample-proportional weighting?
    -----------------------------------
    A client with more data has had more gradient steps and its weights
    encode a more statistically reliable update — its model has seen more
    examples and is less likely to have overfit to noise.  So we give it
    a larger share of the aggregate.

    The formula is:
        w_global = Σ_k  (n_k / N) × w_k

    where n_k is the number of training samples on client k and N = Σ n_k.

    The weakness in our non-IID setup:
    ------------------------------------
    CTU-13 is much larger than SUEE-8.  With this weighting, CTU-13
    dominates every aggregation round.  The result is a global model that
    is essentially a CTU-13 model with minimal influence from the others.
    FedProx mitigates this on the client side (by constraining drift),
    but the aggregation itself remains the same.

    Parameters
    ----------
    client_weights : list of weight lists, one per client
    client_sizes   : number of training samples per client (same order)

    Returns
    -------
    aggregated weights as a list of tensors
    """
    total = sum(client_sizes)
    num_layers = len(client_weights[0])

    aggregated = []
    for layer_idx in range(num_layers):
        layer_avg = torch.zeros_like(client_weights[0][layer_idx])
        for client_idx, weights in enumerate(client_weights):
            fraction = client_sizes[client_idx] / total
            layer_avg += fraction * weights[layer_idx]
        aggregated.append(layer_avg)

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FedNova aggregation
# ─────────────────────────────────────────────────────────────────────────────

def fednova_aggregate(
    global_weights:       List[torch.Tensor],
    client_weights:       List[List[torch.Tensor]],
    client_sizes:         List[int],
    client_local_steps:   List[int],
) -> List[torch.Tensor]:
    """
    FedNova: normalises each client's gradient update by its number of
    local gradient steps before aggregating.

    The problem FedNova solves:
    ----------------------------
    In FedAvg/FedProx, all clients train for the same number of EPOCHS.
    But one epoch on CTU-13 (large) involves far more gradient steps than
    one epoch on SUEE-8 (small).  The client with more steps has moved its
    weights further from the global starting point, even if we think of it
    as "the same number of epochs."

    When FedAvg averages these updates, the client that took more steps
    contributes a proportionally larger weight update, which compounds the
    dominance problem from the sample-size weighting.

    FedNova corrects this: instead of averaging the weights directly, it
    computes the "gradient" each client took (w_local − w_global), divides
    by the number of steps, takes the weighted average of those normalised
    gradients, and then applies the result to the global model.

    Effectively this scales each client's contribution to a "one step"
    equivalent before combining them.

    Parameters
    ----------
    global_weights      : weights at the start of this round (the anchor)
    client_weights      : weights after local training, one list per client
    client_sizes        : training sample counts per client
    client_local_steps  : actual gradient steps taken per client this round
                          (= batches_per_epoch × epochs_completed)
    """
    total = sum(client_sizes)
    num_layers = len(global_weights)

    # Compute normalised gradients: Δw_k / τ_k
    # where Δw_k = w_k_local − w_global and τ_k = local steps
    normalised_deltas = []
    for client_idx, (weights, steps) in enumerate(
        zip(client_weights, client_local_steps)
    ):
        delta = []
        for layer_idx in range(num_layers):
            raw_delta = weights[layer_idx] - global_weights[layer_idx]
            delta.append(raw_delta / max(steps, 1))   # normalise by step count
        normalised_deltas.append(delta)

    # Weighted average of normalised deltas, then apply to global weights
    aggregated = []
    for layer_idx in range(num_layers):
        avg_delta = torch.zeros_like(global_weights[layer_idx])
        for client_idx, deltas in enumerate(normalised_deltas):
            fraction = client_sizes[client_idx] / total
            avg_delta += fraction * deltas[layer_idx]

        # Add the averaged normalised update back to the global weights.
        # The scaling by mean(τ_k) re-introduces a "typical step count"
        # so the global model moves a meaningful distance each round.
        mean_steps = sum(client_local_steps) / len(client_local_steps)
        aggregated.append(global_weights[layer_idx] + mean_steps * avg_delta)

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full federated training loop
# ─────────────────────────────────────────────────────────────────────────────

def federated_train(
    clients:        Dict[str, dict],
    global_model:   ResidualMLP,
    device:         torch.device,
    num_rounds:     int   = 20,
    local_epochs:   int   = 5,
    lr:             float = 1e-3,
    patience:       int   = 5,
    proximal_mu:    float = 0.0,
    algorithm:      str   = "fedavg",   # "fedavg", "fedprox", "fednova"
    eval_every:     int   = 5,
    verbose:        bool  = True,
) -> Tuple[ResidualMLP, Dict]:
    """
    Main federated training loop.

    Each round:
    1. Broadcast the current global model to all clients.
    2. Each client trains locally for `local_epochs` epochs.
    3. Collect client weights (and step counts if FedNova).
    4. Aggregate into a new global model.
    5. Every `eval_every` rounds, evaluate the global model on each
       client's test set to monitor per-client generalisation.

    Parameters
    ----------
    clients     : dict from data.load_all_clients()
    global_model: the model to train (initialised before calling this)
    device      : cpu or cuda
    num_rounds  : total FL communication rounds
    local_epochs: max epochs per client per round (early stopping may
                  reduce this)
    lr          : learning rate for local Adam optimisers
    patience    : early stopping patience for local training
    proximal_mu : FedProx μ parameter (0.0 = FedAvg, try 0.01, 0.1, 1.0)
    algorithm   : which aggregation to use
    eval_every  : evaluate on test sets every N rounds (expensive, do
                  not set to 1 for production runs)
    verbose     : print round summaries

    Returns
    -------
    global_model : trained global model
    history      : dict with per-round metrics for plotting
    """
    assert algorithm in ("fedavg", "fedprox", "fednova"), \
        f"Unknown algorithm '{algorithm}'. Choose fedavg, fedprox, or fednova."

    # Use "fedprox" as a flag to set proximal_mu if the user forgot
    if algorithm == "fedprox" and proximal_mu == 0.0:
        proximal_mu = 0.01
        print(f"  [Warning] algorithm=fedprox but proximal_mu=0. "
              f"Setting proximal_mu=0.01 automatically.")

    history = {
        "round_train_losses": [],   # mean train loss across clients per round
        "per_client_metrics": {},   # {round: {client: metrics}}
    }

    for rnd in range(1, num_rounds + 1):
        if verbose:
            print(f"\n{'═'*55}")
            print(f"  ROUND {rnd}/{num_rounds}  [{algorithm.upper()}  μ={proximal_mu}]")
            print(f"{'═'*55}")

        global_weights = get_weights(global_model)
        client_weights_list = []
        client_sizes        = []
        client_step_counts  = []   # only used by FedNova

        round_losses = []

        for client_name, client_data in clients.items():
            # ── Step 1: send global model to client ──────────────────────────
            local_model = copy.deepcopy(global_model)
            local_model = local_model.to(device)

            # ── Step 2: local training ────────────────────────────────────────
            trained_model, local_history = local_train(
                model         = local_model,
                train_loader  = client_data["train"],
                val_loader    = client_data["val"],
                pos_weight    = client_data["pos_weight"],
                device        = device,
                max_epochs    = local_epochs,
                lr            = lr,
                patience      = patience,
                proximal_mu   = proximal_mu if algorithm in ("fedprox", "fednova") else 0.0,
                verbose       = verbose,
                client_name   = client_name,
            )

            # ── Step 3: collect weights ───────────────────────────────────────
            client_weights_list.append(get_weights(trained_model))

            # Number of training samples this client has
            n_samples = len(client_data["train"].dataset)
            client_sizes.append(n_samples)

            # For FedNova: count actual gradient steps taken
            epochs_done = len(local_history["train_loss"])
            steps_per_epoch = len(client_data["train"])
            client_step_counts.append(epochs_done * steps_per_epoch)

            avg_train_loss = sum(local_history["train_loss"]) / epochs_done
            round_losses.append(avg_train_loss)

        # ── Step 4: server aggregation ────────────────────────────────────────
        if algorithm == "fednova":
            new_global_weights = fednova_aggregate(
                global_weights, client_weights_list,
                client_sizes, client_step_counts
            )
        else:
            # Both "fedavg" and "fedprox" use the same aggregation;
            # the difference was in how clients trained (proximal_mu)
            new_global_weights = fedavg_aggregate(
                client_weights_list, client_sizes
            )

        set_weights(global_model, new_global_weights)

        # Record mean training loss for this round
        mean_loss = sum(round_losses) / len(round_losses)
        history["round_train_losses"].append(mean_loss)

        if verbose:
            print(f"\n  → Round {rnd} complete.  Mean train loss: {mean_loss:.4f}")

        # ── Step 5: periodic evaluation on all test sets ─────────────────────
        if rnd % eval_every == 0:
            if verbose:
                print(f"\n  ── Test evaluation at round {rnd} ──")
            round_metrics = {}
            for client_name, client_data in clients.items():
                m = evaluate(
                    global_model, client_data["test"], device,
                    client_name=client_name, verbose=verbose
                )
                round_metrics[client_name] = m
            history["per_client_metrics"][rnd] = round_metrics

    return global_model, history


# ─────────────────────────────────────────────────────────────────────────────
# 5.  μ sweep helper for FedProx
# ─────────────────────────────────────────────────────────────────────────────

def fedprox_mu_sweep(
    clients:      Dict[str, dict],
    device:       torch.device,
    mu_values:    List[float] = None,
    num_rounds:   int   = 10,
    local_epochs: int   = 5,
    input_dim:    int   = 31,
) -> Dict:
    """
    Trains with FedProx at multiple μ values and returns the results.

    Why sweep μ?
    ------------
    μ controls the trade-off between local adaptation and global coherence.
    Too small (≈0): clients drift freely, similar to FedAvg, no benefit.
    Too large (≈1+): clients barely move from the global model each round,
    slow convergence, poor local adaptation.
    The right μ is dataset-specific, so we test a range and pick the value
    that gives the best balance of F1 scores across all four clients.

    We suggest μ ∈ {0.001, 0.01, 0.1, 1.0} as a practical sweep range.
    """
    if mu_values is None:
        mu_values = [0.001, 0.01, 0.1, 1.0]

    results = {}
    for mu in mu_values:
        print(f"\n{'#'*55}")
        print(f"  FedProx sweep  μ = {mu}")
        print(f"{'#'*55}")

        # Fresh model for each μ value so results are comparable
        model = ResidualMLP(input_dim=input_dim)
        trained_model, history = federated_train(
            clients       = clients,
            global_model  = model,
            device        = device,
            num_rounds    = num_rounds,
            local_epochs  = local_epochs,
            proximal_mu   = mu,
            algorithm     = "fedprox",
            eval_every    = num_rounds,   # only evaluate at the end
            verbose       = False,
        )
        # Final evaluation
        final_metrics = {}
        for client_name, client_data in clients.items():
            final_metrics[client_name] = evaluate(
                trained_model, client_data["test"], device,
                client_name=f"μ={mu} | {client_name}", verbose=True
            )
        results[mu] = final_metrics

    return results
