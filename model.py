"""
model.py  —  Residual MLP for Network Intrusion Detection
==========================================================
Architecture: 31 features → [64 → 128 → 64] residual blocks → sigmoid output
One file, fully self-contained. Import ResidualMLP wherever you need it.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A single residual block: Linear → BN → ReLU → Dropout → Linear → BN
    with a skip connection that adds the block's input to its output.

    Why this structure?
    -------------------
    In standard layers, if the gradient becomes very small in a deep layer
    (vanishing gradient), that layer barely updates during training. In our
    federated setup each client runs several local epochs before sending its
    weights to the server, so deep layers could easily stagnate on some
    clients while updating normally on others. When we then average those
    inconsistent models, the result is incoherent.

    The skip connection creates a direct highway for gradients: even if the
    learned transformation f(x) produces tiny gradients, the identity path
    (x itself) carries the gradient cleanly backwards through the whole
    network, keeping every layer updating consistently on every client.

    Projection:
    -----------
    If in_features != out_features the skip connection can't be a plain
    addition because the shapes don't match.  We project with a 1×1 linear
    (no bias, no activation) to match dimensions.  This is the standard
    approach from the ResNet paper.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),   # normalises activations → scale-invariant
            nn.ReLU(),
            nn.Dropout(dropout),            # randomly zeros 30% of neurons each step
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )

        # Projection layer: only used when input and output dimensions differ
        self.projection = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # f(x) is what the block learned; x is the identity path
        return self.relu(self.block(x) + self.projection(x))


class ResidualMLP(nn.Module):
    """
    Full binary classifier: 31 input features → 3 residual blocks → sigmoid.

    Layer design (why these sizes):
    --------------------------------
    Layer 1  (31 → 64):  compression.
        Takes the 31 raw statistics and learns a compact representation.
        Features that tend to move together — like packet rate and byte
        count — get jointly encoded rather than treated independently.
        64 is wide enough to capture the main patterns without exploding
        the parameter count.

    Layer 2  (64 → 128):  interaction learning.
        Expands the representation to give the model room to learn complex
        feature combinations.  This is where the model figures out that
        "high packet rate AND small packet size AND low inter-arrival
        variance" is a UDP flood signature, versus "moderate rate AND
        slowly growing connection count" being a slow DoS pattern.

    Layer 3  (128 → 64):  refinement.
        Compresses the rich interactions from layer 2 back down into a
        decision-ready representation.  The bottleneck-expand-contract
        (64→128→64) shape is a deliberate design: it forces information
        to flow through a narrow channel twice, which acts as implicit
        regularisation.

    Output  (64 → 1, sigmoid):
        Produces P(attack | features) ∈ [0, 1].  The sigmoid is the
        correct final activation for binary cross-entropy loss and gives
        us the well-calibrated probability score we need for the
        human-in-the-loop thresholds.
    """

    def __init__(self, input_dim: int = 31, dropout: float = 0.3):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualBlock(input_dim, 64,  dropout),   # compression
            ResidualBlock(64,        128, dropout),   # interaction
            ResidualBlock(128,       64,  dropout),   # refinement
        )

        self.head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.layers(x)).squeeze(1)   # shape: (batch,)


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ResidualMLP()
    dummy  = torch.randn(16, 31)          # 16 samples, 31 features
    output = model(dummy)
    print(f"Output shape : {output.shape}")          # should be torch.Size([16])
    print(f"Output range : [{output.min():.3f}, {output.max():.3f}]")  # should be in (0,1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")    # should be ~25 000
