# MHD Framework

Explicit hypergraph computation for PyTorch, maintained by Haoding Souray Meng.

This initial public package includes:
- Tensor-state nodes, operation edges, ordered topology and graph execution
- A named sequential graph adapter with feature endpoints
- ResNet adapters for 2D inputs and explicitly inflated 3D inputs
- Synthetic CPU tests for node aggregation, gradients, graph endpoints and state recovery

## Install

Use Python 3.11–3.13 with PyTorch 2.8 or later.

    python -m pip install -e .
    python -m pip install -e '.[models,dev]'

The models extra installs torchvision for ResNet construction. The core and named
synthetic graph example need only PyTorch. No trained weights are included.

## Quick example

    python examples/synthetic_graph.py

The example builds two linear operations, runs random CPU tensors through named
feature endpoints, and computes gradients using PyTorch autograd.

The core imports are:

    from mhd_framework import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph

Named graphs and ResNet construction are available through:

    from mhd_framework.models import NamedModelGraph, ResNetConfig, create_model

Graph execution uses explicit forward and backward levels. ResNet models provide
forward_until, forward_from, describe_nodes and native state-dictionary methods.

## Tests

    python -m pytest

These tests use synthetic tensors and temporary in-memory state. Full architecture,
GPU, distributed execution and application performance are outside this package's
test coverage. Trainer utilities and additional architecture families are not
included in this initial public package.

## License

MIT. See LICENSE for the full notice and NOTICE.md for dependency attribution.
