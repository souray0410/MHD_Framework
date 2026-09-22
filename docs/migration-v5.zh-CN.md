# V4 → V5 迁移

Message 仍统一为 Tensor；Node、Edge、Topo、Graph、导入路径和显式 levels
执行方式保留。本次不引入字典状态、动态选路、高阶求导或新优化算法。

## 建图及反传

V4 中仅有一个输入运算的节点，`aggregation="replace"` 改为
`aggregation="sum", memory=False`。多个输入运算时必须按数学意图选择聚合；
最后一次写入与求和不等价。`memory` 只控制聚合是否包含旧状态。

V5 对选定前向记录计算原生一阶 VJP。仅实数单元素终点默认反向输入为 1；
向量和复数终点需要在前后向前显式设置 `node.gradient_message.initial_state`。
显式零会保留其“已配置”身份，不等于未配置的默认零。日志中的输出均值不定义
优化目标。自定义 PyTorch 训练循环继续使用；不要求项目改用 Trainer。

## 断点

每次成功完成 `train_step` 后均可调用
`trainer.save_checkpoint(epoch, data_cursor=cursor)`，包括未完成的梯度累积窗口。
重建相同图、优化器、scheduler 和 Trainer 配置后调用
`trainer.load_checkpoint(epoch=epoch)`；调用方从 `trainer.data_cursor` 取回并恢复
数据迭代、采样和增强位置。Trainer 不推测外部数据游标。

唯一当前格式 `mhd_trainer_v5_step_1` 保存待累积梯度、窗口位置及除数、优化器、
scheduler、scaler、节点值、显式 cotangent 标记、Python/NumPy/Torch 及当前设备
CUDA 随机状态，保留 `grad=None`。不序列化未结束的计算图。分布式保存恢复须由
所有 rank 参与；原生续训要求相同并行布局和精度。更换布局或硬件须另外迁移验收。
Inferencer 可将规范模型状态集中加载至单设备，并设置 eval 模式。仅加载可信断点。

当前 Trainer 加载器不回退读取 V4 或旧 V5 预览格式。独立的一次性转换工具负责
来源清单、映射及验收。缺少优化器、随机流或数据位置时，不能声称精确续训。
选优权重与完整断点必须区分；保留原训练历史和源文件。即使 Tensor 数值不变，
序列化文件变化也要登记新 SHA。

## 并行

每次只使用 DDP、FSDP2、显式模块 TP、PP 中一族。PP 准备时执行实际 autocast
微批路径，确定通信 shape/dtype 并恢复 buffers 和 RNG。输入 batch 必须非空、
大小一致、可被 microbatch 数整除。BatchNorm、跨样本目标和随机层必须对照相同
微批计算；不能仅凭有效 batch 相同认为等价。

GPipe、1F1B 用 `pipeline_loss_fn` 定义标量目标，参数和输入梯度均使用正确的
微批归一化；FP16 溢出决定跨 stage 同步。PP 仅在完整 schedule 后保存，要求
`grad_accum_steps=1`，不支持 retain_graph 和显式终点 cotangent。共享参数的所有
使用必须位于同一 stage；跨 stage 共享会报错，避免各 stage 用部分梯度独立更新。

## 模型和验收

安装 `models` 或 `models_extended` 可选依赖后使用
`mhd_framework.models.create_model`。迁入的架构保留 V4 节点、分支、特征端点
和参数共享。当前资产加载要求已转换验收的 V5 产物；数据与研究训练配方归原项目。

对照相同权重和输入，检查输出、输入及参数梯度、buffers、连续更新、严格重载和
续训。FP32 默认 `rtol=1e-5, atol=1e-6`，混合精度默认
`rtol=2e-2, atol=2e-3`。架构测试通过不代表所有生产资产已迁移。
