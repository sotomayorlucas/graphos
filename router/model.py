"""RouterGraph MLP classifier — the core GraphOS computation unit."""

import torch
import torch.nn as nn

from core.constants import TENSOR_DIM, HIDDEN_DIM, NUM_CLASSES


class RouterGraph(nn.Module):
    """3-layer MLP that classifies 64-byte packet tensors into HTTP/DNS/OTHER.

    Intentionally small (~3,264 parameters) for NPU speed.
    Outputs raw logits (no softmax) — use argmax at inference.
    """

    def __init__(self, input_dim=TENSOR_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
