#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MASTER_ADDR:-}" || -z "${MASTER_PORT:-}" || -z "${NNODES:-}" || -z "${NODE_RANK:-}" || -z "${GPUS_PER_NODE:-}" ]]; then
  echo "Usage:" >&2
  echo "  MASTER_ADDR=<ip> MASTER_PORT=<port> NNODES=<n> NODE_RANK=<rank> GPUS_PER_NODE=<gpus> \\" >&2
  echo "  [BATCH_SIZE=<per_gpu_batch>] $0 [extra train.py args]" >&2
  exit 1
fi

TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
BATCH_SIZE=${BATCH_SIZE:-4}

EXTRA_ARGS=("$@")
ADD_BATCH_SIZE=1
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--batch_size" || "$arg" == "--batch_size="* ]]; then
    ADD_BATCH_SIZE=0
    break
  fi
done

CMD=(
  "$TORCHRUN_BIN"
  "--nnodes=$NNODES"
  "--nproc_per_node=$GPUS_PER_NODE"
  "--node_rank=$NODE_RANK"
  "--master_addr=$MASTER_ADDR"
  "--master_port=$MASTER_PORT"
  "train.py"
)

if [[ $ADD_BATCH_SIZE -eq 1 ]]; then
  CMD+=("--batch_size" "$BATCH_SIZE")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
