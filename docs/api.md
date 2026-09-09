# API — V4

```python
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from mhd_framework.utils import MHD_Trainer
```

## State and operations

`MHD_Node.Message` carries initial and current state. Nodes expose feature and gradient messages. `MHD_Edge.Operation` wraps an operation; an edge stores an ordered list of operations, including ordinary PyTorch modules.

## Topology and execution

Role matrices identify sending and receiving nodes for each edge. Sort matrices specify argument and output order. A topology declares execution levels; `graph.forward(levels=[...])` and `graph.backward(levels=[...])` select the sequence explicitly. See [the runnable example](../examples/basic.py) for matrices and a complete graph.

## Version semantics

V4 uses explicit forward/backward level sequences and feature/gradient messages. `replace` is the default aggregation; other built-in aggregations include the current state. Backward levels must match the recorded forward trace.

## Utilities

`utils.py` provides data loading, monitoring, optimizer/training helpers, checkpoint operations and explicit distributed execution utilities. A trainer requires a criteria callable; do not use the retired criteria_node argument. Construct the optimizer from graph parameters and pass the required forward and backward levels. See the unit tests for executable trainer and checkpoint examples.

## Persistence

The package migration preserves tensor operations. Strict state_dict loading, outputs and gradients are checked independently. Full-object Python pickles include module paths and require the original environment when those paths differ. [History](history.md) identifies archived source; this release does not silently alias obsolete module paths.
