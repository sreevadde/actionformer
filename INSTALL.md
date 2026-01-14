# Installation

## Requirements

### Base Requirements
- Linux
- Python 3.8+
- PyTorch 1.11+ (PyTorch 2.0+ required for v2 features)
- CUDA 11.0+
- GCC 4.9+

### Python Dependencies
```
numpy>=1.11,<=1.23
pyyaml
pandas
h5py
joblib
tensorboard
```

### Optional (for v2 Transformer)
- **PyTorch 2.0+**: Required for Flash Attention (SDPA)
- **flash-attn**: Optional, for even faster attention on Ampere+ GPUs

## Installation

### 1. Create Environment
```bash
conda create -n actionformer python=3.10
conda activate actionformer
```

### 2. Install PyTorch
```bash
# For v2 features (Flash Attention, etc.)
pip install torch>=2.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for base features only
pip install torch>=1.11 torchvision
```

### 3. Install Dependencies
```bash
pip install numpy pyyaml pandas h5py joblib tensorboard
```

### 4. Install Flash Attention (Optional)
```bash
# Only for Ampere+ GPUs (A100, RTX 30xx, RTX 40xx)
pip install flash-attn --no-build-isolation
```

### 5. Install ActionFormer
```bash
pip install .
# Or for development
pip install -e .
```

This compiles the NMS extension automatically. Reinstall when PyTorch is updated.

## Feature Compatibility

| Feature | PyTorch Version | Notes |
|---------|-----------------|-------|
| Base ActionFormer | 1.11+ | Original features |
| Flash Attention (SDPA) | 2.0+ | `use_flash_attn: true` |
| Transformer v2 | 2.0+ | RoPE, RMSNorm, SwiGLU |
| Multi-GPU (DDP) | 1.11+ | `torchrun` |
| AMP Training | 1.11+ | `--amp` flag |

## Verification

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "from actionformer import batched_nms; print('NMS compiled successfully')"
```

For v2 features:
```bash
python test_improvements.py
```
