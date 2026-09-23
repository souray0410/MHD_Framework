# MHD Framework

**基于 PyTorch 的显式超图计算框架。**

[API 文档](docs/api.md) · [安装与版本](docs/installation.md) · [版本下载](https://github.com/souray0410/MHD_Framework/releases) · [English](README.md)

Node 保存张量状态，Edge 封装运算，Topo 定义节点与边的连接、顺序和执行层级，Graph 执行指定层级。底层计算和梯度由 PyTorch 模块与 autograd 完成。

## 安装

支持 Python 3.11–3.13、PyTorch 2.8 及以上。先根据硬件安装对应 PyTorch，再选择框架版本：

```bash
python -m pip install "mhd-framework @ git+https://github.com/souray0410/MHD_Framework.git@V5"
```

- **V4**：计算语义冻结的历史研究版本。
- **V5**：Tensor Message 的正式版本；应用权重与续训断点另行迁移验收。

版本源码和 wheel 通过 GitHub Releases 下载，各版本使用相同导入路径：

```python
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from mhd_framework.utils import MHD_Trainer
```

完整可运行示例见 [examples/basic.py](examples/basic.py)。它构建线性层与标量损失组成的超图，执行前向和反向，使用合成 CPU 张量。

开发、测试与目录说明见[英文首页](README.md)和[贡献指南](CONTRIBUTING.md)。[MIT 许可证](LICENSE)。

## models

安装 `models` 可选依赖后，通过 `mhd_framework.models.create_model` 构建带有具名节点端点的模型；扩展编码器使用 `models_extended`。见[可运行模型示例](examples/model.py)和[模型及权重登记规范](docs/models.md)。架构实现、训练完成与权重可复用分别验收。

[V4 → V5 迁移说明](docs/migration-v5.zh-CN.md)列出接口变化、断点及独立转换边界。Message 仍统一为 Tensor。
