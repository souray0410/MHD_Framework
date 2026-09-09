# MHD Framework

**Explicit hypergraph computation for PyTorch.**

[Documentation](docs/api.md) · [Installation and versions](docs/installation.md) · [Releases](https://github.com/souray0410/MHD_Framework/releases) · [中文](README.zh-CN.md)

MHD Framework represents neural computation with four components: nodes carry tensor state, edges wrap operations, topology defines connections and execution levels, and a graph executes the selected levels. PyTorch modules and autograd provide the underlying computation.

## Installation

Python 3.11–3.13 and PyTorch 2.8 or later are required. Install the appropriate PyTorch build for your hardware, then select a framework release:

```bash
python -m pip install "mhd-framework @ git+https://github.com/souray0410/MHD_Framework.git@V4"
```

| Version | Status | Source |
|---|---|---|
| V4 | Stable API with frozen computation semantics | [V4](https://github.com/souray0410/MHD_Framework/tree/V4) |
| V5 | Development preview | [V5](https://github.com/souray0410/MHD_Framework/tree/V5) |
| main | Ongoing V5 development | [main](https://github.com/souray0410/MHD_Framework/tree/main) |

Release archives and wheels are distributed through GitHub Releases. See [installation](docs/installation.md) for preview and editable installs.

## Core API

```python
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from mhd_framework.utils import MHD_Trainer
```

| Component | Responsibility |
|---|---|
| `MHD_Node` | Feature and gradient message state |
| `MHD_Edge` | Ordered operations, including `torch.nn.Module` instances |
| `MHD_Topo` | Node–edge roles, ordering and execution levels |
| `MHD_Graph` | Graph execution and gradient propagation |

The import paths stay consistent across releases; behavior is determined by the installed version. See the [API guide](docs/api.md) for aggregation and backward semantics.

## Example

A complete two-operation graph is provided in [examples/basic.py](examples/basic.py):

```bash
python examples/basic.py
```

The example wraps a linear layer and scalar loss, executes their forward levels, and propagates gradients through the reverse levels. It runs on synthetic CPU tensors.

## Development

```bash
git clone https://github.com/souray0410/MHD_Framework.git
cd MHD_Framework
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/manage.py test
python -m build
```

Source is under `src/mhd_framework/`; tests, examples and documentation have separate directories. [Contributing](CONTRIBUTING.md) describes change and validation requirements. [Release policy](docs/releases.md) describes version stability.

## License

[MIT](LICENSE). Maintained by Haoding Souray Meng.
