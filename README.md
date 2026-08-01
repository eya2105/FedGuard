# FedGuard

FedGuard is a DDoS anomaly-detection project using an autoencoder pipeline, with both centralized and federated training/evaluation workflows.

## Project goal

Build a robust anomaly detection system for network traffic that:
- learns normal behavior,
- detects attack-like deviations,
- and supports federated learning where raw client data stays local.

## Repository overview

- `notebooks/`: full experimentation workflow (data prep, benchmarking, centralized, federated, HiL).
- `scripts/`: script equivalents of notebooks for reproducible non-notebook runs.
- `data/`: input datasets used by experiments.
- `artifacts/models/`: trained model files (centralized and federated).
- `artifacts/outputs/`: plots, metrics, and analysis outputs.
- `docs/`: documentation assets.

## Data used

- `dataset_normal_train.csv`: normal-only traffic windows for unsupervised training.
- `dataset_test_complet.csv`: mixed normal/attack windows for validation and final testing.

## Training flows

### Centralized flow
1. Load normal and mixed datasets.
2. Fit feature transformation on normal training data.
3. Train autoencoder on normal data.
4. Select threshold on validation data.
5. Evaluate once on locked final test split.

### Federated flow
1. Split normal data into temporal client groups.
2. Train local client models starting from global weights.
3. Aggregate client updates with FedAvg on the server.
4. Calibrate global decision threshold.
5. Evaluate on locked final test split.

## Federated client split strategy

Clients are defined by hour ranges to simulate non-IID temporal behavior:

- Client 1: `02:00–09:59`
- Client 2: `10:00–17:59`
- Client 3: `18:00–01:59`

This preserves realistic distribution differences between clients while keeping raw traffic local.

## Autoencoder failure analysis and fix

An early issue was not the model representation itself, but threshold calibration.  
A coarse threshold search over a very wide score range caused unstable classification decisions despite strong score separation.

This was resolved by:
- using a more reliable validation-based thresholding strategy (Youden-J style calibration),
- and keeping a normal-only percentile threshold as a robust fallback reference.

## Result summary

The federated autoencoder reached the same detection behavior as the centralized baseline on the locked evaluation setup, while preserving the federated privacy constraint (no raw data sharing between clients).

