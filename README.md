# MHD Framework

[中文说明](README.zh-CN.md)

A PyTorch toolbox for explicit hypergraph computation. **Node** carries state, **Edge** defines operations, **Topo** specifies connectivity and level order, and **Graph** executes the computation.

**This branch: V5 — development; not used by current LOOK/Radon_Bridge experiments.** The installed release selects the API, as with other Python libraries. There are no bundled parallel historical implementations and no runtime compatibility switch.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

```python
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from mhd_framework.utils import MHD_Trainer
```

This is a source distribution; these instructions do not imply publication on PyPI. Pin an exact Git commit for a reproducible research environment. Version information is available as `mhd_framework.__version__` and `mhd_framework.__api_version__`.

## Versions

- [`main`](https://github.com/souray0410/MHD_Framework): V5 development.
- [`release/v4`](https://github.com/souray0410/MHD_Framework/tree/release/v4): installable V4 with frozen tensor semantics and the same modern project structure.
- [Historical archive](docs/history.md): original version folders, experiments and legacy Python import paths.

Upgrading the installed version is explicit. Separate applications may install different pinned versions in their own environments. An old full-object pickle still requires its original source/environment; do not confuse that with state_dict compatibility.

## Structure

```text
src/mhd_framework/           core.py, utils.py, explicit public package exports
tests/unit/        current release correctness tests
tests/integration/ explicit integration checks
tests/helpers/     synthetic reference models
examples/          minimal runnable examples
benchmarks/        controlled benchmark records
docs/              API, development and history
scripts/           development check/test entry point
```

The toolbox has no cluster account, personal server path, research dataset or job queue. [API guide](docs/api.md) · [Development](docs/development.md) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE).

Haoding Souray Meng (孟号丁) · [souray0410](https://github.com/souray0410)
