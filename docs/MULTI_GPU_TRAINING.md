# Multi-GPU Training Guide

This guide covers distributed training with DistributedDataParallel (DDP), Automatic Mixed Precision (AMP), and gradient accumulation.

## Quick Start

```bash
# 4-GPU training with AMP
torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml --amp --output my_exp
```

## When to Use Multi-GPU Training

| Scenario | Recommendation |
|----------|----------------|
| Single GPU, small dataset | Use `train.py` (original) |
| Single GPU, want faster training | Use `train.py` with AMP manually |
| Multiple GPUs available | Use `train_ddp.py` |
| Limited GPU memory | Use `--accum-steps` for gradient accumulation |
| Large batch size needed | Use DDP + gradient accumulation |
| Production/cluster training | Use DDP with multi-node support |

## DDP Training (`train_ddp.py`)

### Basic Usage

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml --output exp_name

# Single node, 2 GPUs
torchrun --nproc_per_node=2 train_ddp.py configs/thumos_i3d.yaml --output exp_name
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--amp` | False | Enable Automatic Mixed Precision (FP16) |
| `--accum-steps N` | 1 | Gradient accumulation steps |
| `--eval-freq N` | 0 | Validation frequency (0 = disabled) |
| `--ckpt-freq N` | 5 | Checkpoint save frequency (epochs) |
| `--print-freq N` | 10 | Print frequency (iterations) |
| `--resume PATH` | None | Resume from checkpoint |
| `--output NAME` | timestamp | Experiment folder name |

### Examples

```bash
# Fast training with AMP (~2x speedup)
torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml \
    --amp --output fast_training

# Large effective batch size (batch × 4 GPUs × 2 accum = 8x)
torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml \
    --accum-steps 2 --output large_batch

# With periodic validation
torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml \
    --amp --eval-freq 5 --output with_validation

# Resume training
torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml \
    --amp --resume ckpt/exp/epoch_010.pth.tar --output resumed
```

## Automatic Mixed Precision (AMP)

### What is AMP?

AMP uses FP16 (half precision) for most operations while keeping critical operations in FP32. This provides:
- **~2x faster training** on modern GPUs (V100, A100, RTX 30xx/40xx)
- **~50% memory reduction** allowing larger batches
- **Minimal accuracy loss** with proper scaling

### When to Use AMP

| GPU | AMP Benefit | Recommendation |
|-----|-------------|----------------|
| V100, A100, H100 | High (Tensor Cores) | Always use `--amp` |
| RTX 3090, 4090 | High (Tensor Cores) | Always use `--amp` |
| RTX 2080 Ti | Moderate | Use `--amp` |
| GTX 1080 Ti | Low | Optional |
| Older GPUs | Minimal | Skip `--amp` |

### Usage

```bash
torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml --amp
```

## Gradient Accumulation

### What is Gradient Accumulation?

Instead of updating weights after each batch, accumulate gradients over N batches before updating. This simulates a larger batch size without requiring more GPU memory.

**Effective batch size** = `batch_size × num_gpus × accum_steps`

### When to Use

| Scenario | Accum Steps | Notes |
|----------|-------------|-------|
| Normal training | 1 | Default, no accumulation |
| GPU memory limited | 2-4 | Reduce batch_size, increase accum_steps |
| Want larger batch | 2-8 | Maintains same memory, larger effective batch |
| Research (batch size ablation) | Variable | Test different effective batch sizes |

### Example

```bash
# Config has batch_size=2, want effective batch of 32 on 4 GPUs
# 2 × 4 × 4 = 32
torchrun --nproc_per_node=4 train_ddp.py config.yaml --accum-steps 4
```

## Multi-Node Training

For training across multiple machines:

```bash
# On node 0 (master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.1 --master_port=29500 \
    train_ddp.py configs/thumos_i3d.yaml --amp

# On node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.1 --master_port=29500 \
    train_ddp.py configs/thumos_i3d.yaml --amp
```

## Learning Rate Scaling

DDP automatically scales the learning rate:

```
scaled_lr = base_lr × num_gpus × accum_steps
```

This follows the [linear scaling rule](https://arxiv.org/abs/1706.02677) for distributed training.

## Troubleshooting

### NCCL Errors

```bash
# Set NCCL debug info
export NCCL_DEBUG=INFO

# Use different NCCL algorithm
export NCCL_ALGO=Ring
```

### Out of Memory

1. Reduce `batch_size` in config
2. Increase `--accum-steps`
3. Enable `--amp`

### Slow Training

1. Ensure GPUs are on same PCIe switch
2. Use NVLink if available
3. Check `num_workers` in config (reduce if CPU bottleneck)

## Performance Tips

1. **Always use AMP** on modern GPUs
2. **Pin memory**: Already enabled in train_ddp.py
3. **Optimal workers**: `num_workers = 4 × num_gpus` typically
4. **Batch size**: Larger is generally better (within memory limits)
5. **Gradient accumulation**: Use if memory limited

## Comparison: train.py vs train_ddp.py

| Feature | train.py | train_ddp.py |
|---------|----------|--------------|
| Single GPU | Yes | Yes (1 process) |
| Multi-GPU | No | Yes (DDP) |
| AMP | No | Yes (`--amp`) |
| Grad Accumulation | No | Yes (`--accum-steps`) |
| Distributed Validation | No | Yes (`--eval-freq`) |
| Multi-node | No | Yes |
| Learning rate scaling | Manual | Automatic |
