# MHD Framework

基于 PyTorch 的显式超图计算工具箱。Node 携带状态，Edge 承载运算，Topo 描述连接和执行 Level，Graph 执行计算。

**当前分支提供 V5。** main 维护开发中的 V5，`release/v4` 提供 LOOK 与 Radon_Bridge 使用的 V4。和 PyTorch 一样，安装哪个版本就使用哪个版本；源码不同时装入多代实现，也没有兼容开关。

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

两个发布分支均采用 `src/mhd_framework/core.py`、`src/mhd_framework/utils.py`、tests、docs、examples、scripts 等统一工程结构，计算语义分别由各版本定义。项目在独立环境中锁定具体 Git 提交；安装版本更新不会静默发生。

原始 V1–V5 文件夹、历史实验和旧导入路径完整保存在[归档分支](docs/history.md)。旧整对象 pickle 仍需原环境；不能把 state_dict 可加载解释为旧 pickle 可直接加载。

工具箱不绑定 Ibex、ws02、数据路径或研究任务。[API](docs/api.md) · [开发](docs/development.md) · [贡献](CONTRIBUTING.md) · [版本与目录](README.md)。
