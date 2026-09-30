"""Run a small named graph using only random CPU tensors."""
from dataclasses import dataclass

import torch
from torch import nn

from mhd_framework.models import NamedModelGraph


@dataclass(frozen=True)
class Config:
    name: str = "synthetic_linear_graph"
    input_width: int = 4
    feature_width: int = 3
    output_width: int = 2


def main():
    torch.manual_seed(17)
    config = Config()
    native = nn.Sequential()
    native.add_module("features", nn.Linear(config.input_width, config.feature_width))
    native.add_module("logits", nn.Linear(config.feature_width, config.output_width))
    graph = NamedModelGraph(
        config, native,
        [("features", native.features), ("logits", native.logits)],
        {"features": config.feature_width}, device="cpu",
    )
    inputs = torch.randn(2, config.input_width, requires_grad=True)
    features = graph.forward_until("features", inputs)
    output = graph.forward_from("features", features)
    torch.testing.assert_close(output, native(inputs))
    output.square().mean().backward()
    assert inputs.grad is not None
    print("Output shape:", tuple(output.shape))
    print("Named endpoints:", sorted(graph.endpoint_nodes))


if __name__ == "__main__":
    main()
