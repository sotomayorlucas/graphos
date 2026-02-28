"""Route table as a tensor program — matmul-based packet routing."""

import os

import torch
import torch.nn as nn

from core.constants import (
    TENSOR_DIM, NUM_ROUTES, DEFAULT_BATCH_SIZE,
    OFFSET_PROTOCOL, OFFSET_DST_PORT, OFFSET_TTL,
)
from kernel.program import ProgramSpec


class RouteTableModel(nn.Module):
    """Route table expressed as a single linear layer (matmul).

    Routing = W @ packet -> scores, route_id = argmax(scores).
    4 routes: LOCAL(0), FORWARD(1), DROP(2), MONITOR(3).
    Weights are hand-crafted (not trained) to demonstrate that
    traditional routing logic can be expressed as tensor ops.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(TENSOR_DIM, NUM_ROUTES, bias=False)
        self._encode_rules()

    def _encode_rules(self):
        """Set hand-crafted weights encoding routing rules.

        Byte values are normalized to [0, 1] (raw_byte / 255):
          proto: TCP=6→0.024, UDP=17→0.067, ICMP=1→0.004
          dst_port: DNS 53→(0, 0.208), HTTP 80→(0, 0.314)
          TTL: range 32-128 → 0.125-0.502

        Strategy: use protocol as primary differentiator since
        UDP (0.067) is 2.8x TCP (0.024), with port as secondary signal.
        """
        with torch.no_grad():
            w = torch.zeros(NUM_ROUTES, TENSOR_DIM)

            # LOCAL (0): DNS traffic = UDP protocol
            # 100 * 0.067(UDP) = 6.7 >> 100 * 0.024(TCP) = 2.4
            w[0, OFFSET_PROTOCOL] = 100.0

            # FORWARD (1): TCP + well-known HTTP ports
            # Negative on protocol penalizes UDP more than TCP:
            # -30*0.024(TCP)=-0.7, -30*0.067(UDP)=-2.0
            # Positive on port byte: 25*0.314(80)=7.9, 25*0.208(53)=5.2
            # Net HTTP: -0.7 + 7.9 = 7.2
            # Net DNS:  -2.0 + 5.2 = 3.2 (loses to LOCAL's 6.7)
            w[1, OFFSET_PROTOCOL] = -30.0
            w[1, OFFSET_DST_PORT + 1] = 25.0
            w[1, OFFSET_DST_PORT] = 8.0        # Boost high-port services (8080 etc)

            # DROP (2): low TTL → expired/suspicious packets
            # -20 * 0.5(normal TTL) = -10 + 1.0 baseline = -9 (never wins)
            # -20 * 0.0(zero TTL) = 0 + 1.0 baseline = 1.0 (wins when others ~0)
            w[2, OFFSET_TTL] = -20.0
            w[2, :] += 0.015                    # ~0.015 * 64 * 0.5 = ~0.5 baseline

            # MONITOR (3): ICMP (proto=1→0.004) + packets worth inspecting
            # -100*0.004(ICMP)=-0.4, -100*0.024(TCP)=-2.4, -100*0.067(UDP)=-6.7
            # +12*TTL: 12*0.5=6.0
            # Net ICMP: -0.4+6.0=5.6 (wins)
            # Net TCP:  -2.4+6.0=3.6 (loses to FORWARD's 7.2)
            # Net UDP:  -6.7+6.0=-0.7 (loses to LOCAL's 6.7)
            w[3, OFFSET_PROTOCOL] = -100.0
            w[3, OFFSET_TTL] = 12.0

            self.linear.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, 64) -> (B, 4) route scores."""
        return self.linear(x)


def route_table_spec(
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_dir: str = "models",
) -> ProgramSpec:
    """Create a ProgramSpec for the route table program."""
    filename = f"route_table_b{batch_size}.onnx"
    return ProgramSpec(
        name="route_table",
        onnx_path=os.path.join(model_dir, filename),
        input_shape=(batch_size, TENSOR_DIM),
        output_shape=(batch_size, NUM_ROUTES),
        description="Matmul route table: LOCAL/FORWARD/DROP/MONITOR",
    )
