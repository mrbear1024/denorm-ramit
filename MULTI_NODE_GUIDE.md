Multi-node DDP run guide (train.py)
===================================

This guide describes how to launch multi-machine training with torchrun.

Prerequisites
-------------
- Same code and data paths on each node (or a shared filesystem).
- Network connectivity between nodes; MASTER_PORT open.
- Matching PyTorch/CUDA/NCCL versions if possible.
- Decide how many GPUs per node are used.

Key parameters
--------------
- nnodes: number of machines.
- nproc_per_node: GPUs per machine.
- node_rank: machine index, 0..(nnodes-1).
- master_addr: IP of the master node (rank 0).
- master_port: a free port on the master node.
- batch_size: per-GPU batch size.

Recommended launch (2 nodes example, 8 GPUs per node)
-----------------------------------------------------
On the master node (node_rank=0):

  export MASTER_ADDR=<master_ip>
  export MASTER_PORT=29500
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py --batch_size 4

On the worker node (node_rank=1):

  export MASTER_ADDR=<master_ip>
  export MASTER_PORT=29500
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py --batch_size 4

N nodes template
----------------
Repeat on each node, changing node_rank:

  export MASTER_ADDR=<master_ip>
  export MASTER_PORT=29500
  torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank> \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py --batch_size <per_gpu_batch>

Optional environment tips
-------------------------
- Limit GPUs on a node:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...
- Set NCCL network interface when needed:
  export NCCL_SOCKET_IFNAME=eth0
- If your cluster blocks ports, pick another master_port.

Notes
-----
- Do not pass --gpu when using torchrun; use CUDA_VISIBLE_DEVICES instead.
- Total batch size = per_gpu_batch * nnodes * nproc_per_node.
