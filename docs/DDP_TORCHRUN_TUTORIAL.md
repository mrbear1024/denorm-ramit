# PyTorch DDP + torchrun 入门教程

本教程介绍如何使用 PyTorch 的 DistributedDataParallel (DDP) 和 torchrun 进行分布式训练。

---

## 目录

1. [基本概念](#1-基本概念)
2. [环境变量说明](#2-环境变量说明)
3. [代码改造步骤](#3-代码改造步骤)
4. [启动训练](#4-启动训练)
5. [完整代码示例](#5-完整代码示例)
6. [常见问题](#6-常见问题)

---

## 1. 基本概念

### 什么是 DDP？

**DistributedDataParallel (DDP)** 是 PyTorch 官方推荐的数据并行方案：

- 每个 GPU 运行一个独立进程，持有完整的模型副本
- 数据被分片到各个进程，每个进程处理不同的数据子集
- 梯度通过 AllReduce 操作在进程间同步
- 相比 DataParallel，DDP 没有 GIL 限制，效率更高

### 什么是 torchrun？

**torchrun** 是 PyTorch 提供的分布式启动工具（取代了旧版的 `torch.distributed.launch`）：

- 自动设置环境变量（RANK, WORLD_SIZE, LOCAL_RANK 等）
- 支持单机多卡和多机多卡
- 提供弹性训练能力（容错重启）

### 关键术语

| 术语 | 含义 |
|------|------|
| `world_size` | 总进程数（= 机器数 × 每机器 GPU 数） |
| `rank` | 当前进程的全局编号（0 到 world_size-1） |
| `local_rank` | 当前进程在本机的编号（0 到 nproc_per_node-1） |
| `node_rank` | 当前机器的编号（0 到 nnodes-1） |

**示例**：2 台机器，每台 4 张卡

```
机器0: local_rank=0,1,2,3 → rank=0,1,2,3
机器1: local_rank=0,1,2,3 → rank=4,5,6,7
world_size = 8
```

---

## 2. 环境变量说明

torchrun 会自动设置以下环境变量：

```python
import os

# 必需的环境变量
RANK = os.environ["RANK"]              # 全局进程编号
WORLD_SIZE = os.environ["WORLD_SIZE"]  # 总进程数
LOCAL_RANK = os.environ["LOCAL_RANK"]  # 本机进程编号

# 通信相关
MASTER_ADDR = os.environ["MASTER_ADDR"]  # 主节点 IP
MASTER_PORT = os.environ["MASTER_PORT"]  # 主节点端口
```

---

## 3. 代码改造步骤

### 步骤 1：初始化分布式环境

```python
import os
import torch
import torch.distributed as dist

def init_distributed_mode():
    """初始化分布式训练环境"""
    # 检查是否由 torchrun 启动
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        distributed = True
    else:
        # 单卡模式
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    if distributed:
        # 设置当前进程使用的 GPU
        torch.cuda.set_device(local_rank)

        # 初始化进程组
        dist.init_process_group(
            backend="nccl",      # GPU 用 nccl，CPU 用 gloo
            init_method="env://" # 从环境变量读取配置
        )

        # 同步所有进程
        dist.barrier()

    return rank, world_size, local_rank, distributed
```

### 步骤 2：使用 DistributedSampler

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def create_dataloader(dataset, batch_size, distributed, rank, world_size, shuffle=True):
    """创建支持分布式的 DataLoader"""
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        # 使用 sampler 时，DataLoader 的 shuffle 必须为 False
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 重要！
            sampler=sampler,
            pin_memory=True,
            num_workers=4
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4
        )

    return loader
```

### 步骤 3：包装模型为 DDP

```python
from torch.nn.parallel import DistributedDataParallel as DDP

def wrap_model_ddp(model, local_rank, distributed):
    """将模型包装为 DDP"""
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    return model, device
```

### 步骤 4：训练循环中的注意事项

```python
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, distributed):
    model.train()

    # 重要：每个 epoch 设置不同的随机种子，确保数据打乱
    if distributed and hasattr(dataloader, 'sampler'):
        dataloader.sampler.set_epoch(epoch)

    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 汇总所有进程的 loss（可选）
    if distributed:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor.item() / dist.get_world_size()

    return total_loss
```

### 步骤 5：只在主进程执行的操作

```python
def is_main_process(rank, distributed):
    """判断是否为主进程"""
    return (not distributed) or (rank == 0)

# 使用示例
if is_main_process(rank, distributed):
    # 只在主进程执行：
    # - 打印日志
    # - 保存模型
    # - 写入 TensorBoard
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    torch.save(model.module.state_dict(), "model.pth")  # 注意：用 model.module
```

### 步骤 6：清理分布式环境

```python
def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
```

---

## 4. 启动训练

### 单机单卡（普通方式）

```bash
python train.py --batch_size 8
```

### 单机多卡

```bash
# 使用全部 GPU
torchrun --nproc_per_node=4 train.py --batch_size 8

# 指定使用哪些 GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --batch_size 8
```

### 多机多卡

**机器 0（主节点）：**
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500

torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --batch_size 8
```

**机器 1（工作节点）：**
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500

torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --batch_size 8
```

### 多机单卡

```bash
# 机器 0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 \
    train.py --batch_size 8

# 机器 1
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=29500 \
    train.py --batch_size 8
```

---

## 5. 完整代码示例

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def init_distributed():
    """初始化分布式环境"""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    rank, world_size, local_rank, distributed = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # 创建数据
    dataset = TensorDataset(
        torch.randn(1000, 10),
        torch.randn(1000, 1)
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None
    dataloader = DataLoader(dataset, batch_size=32, shuffle=(sampler is None), sampler=sampler)

    # 训练
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        if distributed:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

    # 保存模型（只在主进程）
    if rank == 0:
        state_dict = model.module.state_dict() if distributed else model.state_dict()
        torch.save(state_dict, "model.pth")

    cleanup()


if __name__ == "__main__":
    main()
```

启动：
```bash
# 单卡
python train.py

# 4 卡
torchrun --nproc_per_node=4 train.py
```

---

## 6. 常见问题

### Q1: 进程卡住不动？

检查：
1. 所有节点是否同时启动
2. `MASTER_ADDR` 和 `MASTER_PORT` 是否正确
3. 防火墙是否开放端口

调试：
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Q2: NCCL 网卡选择问题？

指定网卡：
```bash
export NCCL_SOCKET_IFNAME=eth0  # 使用 eth0 网卡
```

查看网卡：
```bash
ip addr show
```

### Q3: 如何正确保存/加载模型？

保存（去掉 DDP 包装）：
```python
if rank == 0:
    state_dict = model.module.state_dict() if distributed else model.state_dict()
    torch.save(state_dict, "model.pth")
```

加载：
```python
model.load_state_dict(torch.load("model.pth", map_location=device))
if distributed:
    model = DDP(model, device_ids=[local_rank])
```

### Q4: Batch Size 如何计算？

```
有效 batch size = per_gpu_batch_size × world_size

例如：
- 2 台机器，每台 4 卡
- per_gpu_batch_size = 8
- 有效 batch size = 8 × 8 = 64
```

### Q5: 学习率是否需要调整？

一般规则：有效 batch size 增大 N 倍，学习率也增大 N 倍（或使用 warmup）。

```python
base_lr = 0.001
scaled_lr = base_lr * world_size
```

### Q6: 不同进程的随机性？

为不同进程设置不同的种子：
```python
seed = base_seed + rank
torch.manual_seed(seed)
```

---

## 参考资料

- [PyTorch DDP 官方教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun 文档](https://pytorch.org/docs/stable/elastic/run.html)
- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
