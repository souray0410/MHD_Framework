# MHD Framework

**基于 PyTorch 的显式超图计算框架。**

[API 文档](docs/api.md) · [安装与版本](docs/installation.md) · [版本下载](https://github.com/souray0410/MHD_Framework/releases) · [English](README.md)

Node 保存张量状态，Edge 封装运算，Topo 定义节点与边的连接、顺序和执行层级，Graph 执行指定层级。底层计算和梯度由 PyTorch 模块与 autograd 完成。

## 安装

支持 Python 3.11–3.13、PyTorch 2.8 及以上。先根据硬件安装对应 PyTorch，再选择框架版本：

```bash
python -m pip install "mhd-framework @ git+https://github.com/souray0410/MHD_Framework.git@V4"
```

- **V4**：计算语义冻结的稳定 API。
- **V5**：开发预览版，需主动选择；main 继续开发此版本。

版本源码和 wheel 通过 GitHub Releases 下载，各版本使用相同导入路径：

```python
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from mhd_framework.utils import MHD_Trainer
```

完整可运行示例见 [examples/basic.py](examples/basic.py)。它构建线性层与标量损失组成的超图，执行前向和反向，使用合成 CPU 张量。

开发、测试与目录说明见[英文首页](README.md)和[贡献指南](CONTRIBUTING.md)。[MIT 许可证](LICENSE)。

## 模型库

[模型结构与训练权重登记规范](docs/model_zoo.md)。当前初始目录处于规划阶段；结构实现、训练完成与权重可复用分别验收。
