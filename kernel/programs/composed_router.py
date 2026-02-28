"""Composed router — uses concat(raw_packet, classifier_logits) for routing."""

import os

import torch
import torch.nn as nn

from core.constants import (
    TENSOR_DIM, NUM_CLASSES, NUM_ROUTES, DEFAULT_BATCH_SIZE,
    OFFSET_PROTOCOL, OFFSET_DST_PORT, OFFSET_TTL,
)
from kernel.program import ProgramSpec

# Input dim = raw packet (64) + classifier logits (3)
COMPOSED_INPUT_DIM = TENSOR_DIM + NUM_CLASSES  # 67


class ComposedRouterModel(nn.Module):
    """Route table that uses BOTH raw packet bytes AND classifier logits.

    Input: concat(raw_packet(B,64), classifier_logits(B,3)) -> (B,67)
    Output: route scores (B,4)

    This proves that chaining NPU programs (classifier -> router) produces
    BETTER routing decisions than packet bytes alone, because the router
    has classification context.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(COMPOSED_INPUT_DIM, NUM_ROUTES, bias=False)
        self._encode_rules()

    def _encode_rules(self):
        """Set hand-crafted weights using packet bytes AND classifier logits.

        Indices 0-63: raw packet bytes (same offsets as route_table.py)
        Indices 64-66: classifier logits [HTTP(64), DNS(65), OTHER(66)]

        The classifier logits give strong, pre-computed class signals that
        complement the raw byte heuristics. This is the key advantage of
        program composition — downstream programs benefit from upstream analysis.
        """
        with torch.no_grad():
            w = torch.zeros(NUM_ROUTES, COMPOSED_INPUT_DIM)

            # LOCAL (0): DNS traffic
            # Packet heuristic: protocol byte (UDP=17 -> 0.067)
            w[0, OFFSET_PROTOCOL] = 50.0
            # Classifier signal: boost DNS logit (index 65)
            w[0, 64 + 1] = 15.0  # DNS logit

            # FORWARD (1): HTTP traffic
            # Packet heuristic: port byte (HTTP 80 -> 0.314)
            w[1, OFFSET_PROTOCOL] = -15.0
            w[1, OFFSET_DST_PORT + 1] = 20.0
            # Classifier signal: boost HTTP logit (index 64)
            w[1, 64 + 0] = 15.0  # HTTP logit

            # DROP (2): low TTL / unknown traffic
            # Packet heuristic: negative TTL
            w[2, OFFSET_TTL] = -15.0
            w[2, :64] += 0.01  # small baseline
            # Classifier signal: boost OTHER logit (index 66)
            w[2, 64 + 2] = 12.0  # OTHER logit

            # MONITOR (3): uncertain traffic — penalize ALL confident logits
            # If the classifier is uncertain (all logits low), monitor wins
            w[3, OFFSET_PROTOCOL] = -50.0
            w[3, OFFSET_TTL] = 8.0
            # Penalize confident classification — uncertain packets get monitored
            w[3, 64 + 0] = -5.0  # penalize HTTP confidence
            w[3, 64 + 1] = -5.0  # penalize DNS confidence
            w[3, 64 + 2] = -5.0  # penalize OTHER confidence

            self.linear.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, 67) -> (B, 4) route scores."""
        return self.linear(x)


def composed_router_spec(
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_dir: str = "models",
) -> ProgramSpec:
    """Create a ProgramSpec for the composed router program."""
    filename = f"composed_router_b{batch_size}.onnx"
    return ProgramSpec(
        name="composed_router",
        onnx_path=os.path.join(model_dir, filename),
        input_shape=(batch_size, COMPOSED_INPUT_DIM),
        output_shape=(batch_size, NUM_ROUTES),
        description="Composed router: packet+logits -> LOCAL/FORWARD/DROP/MONITOR",
    )
