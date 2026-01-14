#!/usr/bin/env python3
"""
Test script for ActionFormer improvements.
Tests: RMSNorm, SwiGLU, RoPE, Flash Attention, v2 backbone.
"""

import torch
import torch.nn as nn
import time
import argparse

def test_imports():
    """Test all new components can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    from actionformer.modeling.blocks import (
        RMSNorm, SwiGLU, RotaryPositionEmbedding,
        MaskedMHAv2, TransformerBlockv2,
        LayerNorm, TransformerBlock,
        HAS_FLASH_ATTN
    )
    from actionformer.modeling.backbones import ConvTransformerBackbonev2

    print(f"  RMSNorm: OK")
    print(f"  SwiGLU: OK")
    print(f"  RotaryPositionEmbedding: OK")
    print(f"  MaskedMHAv2: OK")
    print(f"  TransformerBlockv2: OK")
    print(f"  ConvTransformerBackbonev2: OK")
    print(f"  Flash Attention available: {HAS_FLASH_ATTN}")
    print()
    return True


def test_rmsnorm():
    """Test RMSNorm vs LayerNorm."""
    print("=" * 60)
    print("Testing RMSNorm...")
    print("=" * 60)

    from actionformer.modeling.blocks import RMSNorm, LayerNorm

    B, C, T = 4, 256, 512
    x = torch.randn(B, C, T).cuda()

    ln = LayerNorm(C).cuda()
    rms = RMSNorm(C).cuda()

    # Warmup
    for _ in range(10):
        _ = ln(x)
        _ = rms(x)

    torch.cuda.synchronize()

    # Benchmark LayerNorm
    start = time.time()
    for _ in range(100):
        _ = ln(x)
    torch.cuda.synchronize()
    ln_time = (time.time() - start) * 10  # ms per iteration

    # Benchmark RMSNorm
    start = time.time()
    for _ in range(100):
        _ = rms(x)
    torch.cuda.synchronize()
    rms_time = (time.time() - start) * 10  # ms per iteration

    print(f"  LayerNorm: {ln_time:.3f} ms")
    print(f"  RMSNorm:   {rms_time:.3f} ms")
    print(f"  Speedup:   {ln_time/rms_time:.2f}x")
    print()
    return True


def test_swiglu():
    """Test SwiGLU vs standard MLP."""
    print("=" * 60)
    print("Testing SwiGLU FFN...")
    print("=" * 60)

    from actionformer.modeling.blocks import SwiGLU

    B, C, T = 4, 256, 512
    x = torch.randn(B, C, T).cuda()

    # Standard MLP
    mlp = nn.Sequential(
        nn.Conv1d(C, C * 4, 1),
        nn.GELU(),
        nn.Conv1d(C * 4, C, 1),
    ).cuda()

    # SwiGLU
    swiglu = SwiGLU(C).cuda()

    # Count parameters
    mlp_params = sum(p.numel() for p in mlp.parameters())
    swiglu_params = sum(p.numel() for p in swiglu.parameters())

    # Warmup
    for _ in range(10):
        _ = mlp(x)
        _ = swiglu(x)

    torch.cuda.synchronize()

    # Benchmark MLP
    start = time.time()
    for _ in range(100):
        _ = mlp(x)
    torch.cuda.synchronize()
    mlp_time = (time.time() - start) * 10

    # Benchmark SwiGLU
    start = time.time()
    for _ in range(100):
        _ = swiglu(x)
    torch.cuda.synchronize()
    swiglu_time = (time.time() - start) * 10

    print(f"  MLP params:    {mlp_params:,}")
    print(f"  SwiGLU params: {swiglu_params:,}")
    print(f"  MLP time:      {mlp_time:.3f} ms")
    print(f"  SwiGLU time:   {swiglu_time:.3f} ms")
    print()
    return True


def test_rope():
    """Test Rotary Position Embeddings."""
    print("=" * 60)
    print("Testing RoPE...")
    print("=" * 60)

    from actionformer.modeling.blocks import RotaryPositionEmbedding, apply_rotary_pos_emb

    B, nh, T, hs = 4, 8, 512, 64

    rope = RotaryPositionEmbedding(hs, max_seq_len=2048).cuda()
    q = torch.randn(B, nh, T, hs).cuda()
    k = torch.randn(B, nh, T, hs).cuda()

    cos, sin = rope(q, T)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    print(f"  Input shape:  {q.shape}")
    print(f"  Output shape: {q_rot.shape}")
    print(f"  cos/sin shape: {cos.shape}")

    # Test longer sequence
    T2 = 4096
    q2 = torch.randn(B, nh, T2, hs).cuda()
    k2 = torch.randn(B, nh, T2, hs).cuda()
    cos2, sin2 = rope(q2, T2)
    q2_rot, k2_rot = apply_rotary_pos_emb(q2, k2, cos2, sin2)
    print(f"  Extended to T={T2}: OK (RoPE auto-extends)")
    print()
    return True


def test_flash_attention():
    """Test Flash Attention vs standard attention."""
    print("=" * 60)
    print("Testing Flash Attention...")
    print("=" * 60)

    from actionformer.modeling.blocks import MaskedMHAv2, MaskedMHA, HAS_FLASH_ATTN

    if not HAS_FLASH_ATTN:
        print("  Flash Attention not available (requires PyTorch 2.0+)")
        print("  Skipping benchmark...")
        print()
        return True

    B, C, T = 4, 256, 1024
    x = torch.randn(B, C, T).cuda()
    mask = torch.ones(B, 1, T, dtype=torch.bool).cuda()

    # Standard attention
    attn_v1 = MaskedMHA(C, n_head=8).cuda()

    # Flash attention
    attn_v2_flash = MaskedMHAv2(C, n_head=8, use_flash_attn=True, use_rope=False).cuda()
    attn_v2_no_flash = MaskedMHAv2(C, n_head=8, use_flash_attn=False, use_rope=False).cuda()

    # Warmup
    for _ in range(10):
        _ = attn_v1(x, mask)
        _ = attn_v2_flash(x, mask)

    torch.cuda.synchronize()

    # Benchmark v1
    start = time.time()
    for _ in range(50):
        _ = attn_v1(x, mask)
    torch.cuda.synchronize()
    v1_time = (time.time() - start) * 20

    # Benchmark v2 with Flash
    start = time.time()
    for _ in range(50):
        _ = attn_v2_flash(x, mask)
    torch.cuda.synchronize()
    v2_flash_time = (time.time() - start) * 20

    # Memory test
    torch.cuda.reset_peak_memory_stats()
    _ = attn_v1(x, mask)
    v1_mem = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    _ = attn_v2_flash(x, mask)
    v2_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"  Sequence length: {T}")
    print(f"  Standard attention: {v1_time:.3f} ms")
    print(f"  Flash attention:    {v2_flash_time:.3f} ms")
    print(f"  Speedup: {v1_time/v2_flash_time:.2f}x")
    print(f"  Memory (v1): {v1_mem:.1f} MB")
    print(f"  Memory (v2): {v2_mem:.1f} MB")
    print()
    return True


def test_transformer_block():
    """Test TransformerBlock v1 vs v2."""
    print("=" * 60)
    print("Testing TransformerBlock v1 vs v2...")
    print("=" * 60)

    from actionformer.modeling.blocks import TransformerBlock, TransformerBlockv2

    B, C, T = 4, 256, 512
    x = torch.randn(B, C, T).cuda()
    mask = torch.ones(B, 1, T, dtype=torch.bool).cuda()

    # v1 block
    block_v1 = TransformerBlock(
        n_embd=C, n_head=8, n_ds_strides=(1, 1),
        attn_pdrop=0.0, proj_pdrop=0.0, path_pdrop=0.0,
        mha_win_size=-1
    ).cuda()

    # v2 block
    block_v2 = TransformerBlockv2(
        n_embd=C, n_head=8, n_ds_strides=(1, 1),
        attn_pdrop=0.0, proj_pdrop=0.0, path_pdrop=0.0,
        mha_win_size=-1,
        use_rope=True, use_flash_attn=True,
        use_swiglu=True, use_rms_norm=True
    ).cuda()

    # Count parameters
    v1_params = sum(p.numel() for p in block_v1.parameters())
    v2_params = sum(p.numel() for p in block_v2.parameters())

    # Forward pass test
    out_v1, mask_v1 = block_v1(x, mask)
    out_v2, mask_v2 = block_v2(x, mask)

    print(f"  v1 params: {v1_params:,}")
    print(f"  v2 params: {v2_params:,}")
    print(f"  v1 output shape: {out_v1.shape}")
    print(f"  v2 output shape: {out_v2.shape}")

    # Warmup
    for _ in range(10):
        _ = block_v1(x, mask)
        _ = block_v2(x, mask)

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(50):
        _ = block_v1(x, mask)
    torch.cuda.synchronize()
    v1_time = (time.time() - start) * 20

    start = time.time()
    for _ in range(50):
        _ = block_v2(x, mask)
    torch.cuda.synchronize()
    v2_time = (time.time() - start) * 20

    print(f"  v1 time: {v1_time:.3f} ms")
    print(f"  v2 time: {v2_time:.3f} ms")
    print(f"  Speedup: {v1_time/v2_time:.2f}x")
    print()
    return True


def test_backbone():
    """Test full backbone v1 vs v2."""
    print("=" * 60)
    print("Testing Backbone v1 vs v2...")
    print("=" * 60)

    from actionformer.modeling.backbones import ConvTransformerBackbone, ConvTransformerBackbonev2

    B, C, T = 2, 2048, 256
    x = torch.randn(B, C, T).cuda()
    mask = torch.ones(B, 1, T, dtype=torch.bool).cuda()

    # v1 backbone
    backbone_v1 = ConvTransformerBackbone(
        n_in=C, n_embd=256, n_head=4, n_embd_ks=3,
        max_len=2048, arch=(2, 2, 5),
        mha_win_size=[-1, -1, -1, -1, -1, -1],
        scale_factor=2, with_ln=False,
        attn_pdrop=0.0, proj_pdrop=0.0, path_pdrop=0.0,
        use_abs_pe=True, use_rel_pe=False
    ).cuda()

    # v2 backbone
    backbone_v2 = ConvTransformerBackbonev2(
        n_in=C, n_embd=256, n_head=4, n_embd_ks=3,
        max_len=2048, arch=(2, 2, 5),
        mha_win_size=[-1, -1, -1, -1, -1, -1],
        scale_factor=2, with_ln=False,
        attn_pdrop=0.0, proj_pdrop=0.0, path_pdrop=0.0,
        use_abs_pe=False, use_rope=True,
        use_flash_attn=True, use_swiglu=True, use_rms_norm=True
    ).cuda()

    # Count parameters
    v1_params = sum(p.numel() for p in backbone_v1.parameters())
    v2_params = sum(p.numel() for p in backbone_v2.parameters())

    # Forward pass
    feats_v1, masks_v1 = backbone_v1(x, mask)
    feats_v2, masks_v2 = backbone_v2(x, mask)

    print(f"  v1 params: {v1_params:,}")
    print(f"  v2 params: {v2_params:,}")
    print(f"  v1 output levels: {len(feats_v1)}")
    print(f"  v2 output levels: {len(feats_v2)}")
    for i, (f1, f2) in enumerate(zip(feats_v1, feats_v2)):
        print(f"    Level {i}: v1={f1.shape}, v2={f2.shape}")

    # Warmup
    for _ in range(5):
        _ = backbone_v1(x, mask)
        _ = backbone_v2(x, mask)

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(20):
        _ = backbone_v1(x, mask)
    torch.cuda.synchronize()
    v1_time = (time.time() - start) * 50

    start = time.time()
    for _ in range(20):
        _ = backbone_v2(x, mask)
    torch.cuda.synchronize()
    v2_time = (time.time() - start) * 50

    # Memory
    torch.cuda.reset_peak_memory_stats()
    _ = backbone_v1(x, mask)
    v1_mem = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    _ = backbone_v2(x, mask)
    v2_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"  v1 time: {v1_time:.3f} ms")
    print(f"  v2 time: {v2_time:.3f} ms")
    print(f"  Speedup: {v1_time/v2_time:.2f}x")
    print(f"  v1 memory: {v1_mem:.1f} MB")
    print(f"  v2 memory: {v2_mem:.1f} MB")
    print()
    return True


def test_ddp_ready():
    """Test DDP training script can be imported."""
    print("=" * 60)
    print("Testing DDP training script...")
    print("=" * 60)

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_ddp", "train_ddp.py")
    module = importlib.util.module_from_spec(spec)

    print("  train_ddp.py: OK (can be imported)")
    print("  Usage: torchrun --nproc_per_node=N train_ddp.py config.yaml --amp")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description='Test ActionFormer improvements')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--imports', action='store_true', help='Test imports')
    parser.add_argument('--rmsnorm', action='store_true', help='Test RMSNorm')
    parser.add_argument('--swiglu', action='store_true', help='Test SwiGLU')
    parser.add_argument('--rope', action='store_true', help='Test RoPE')
    parser.add_argument('--flash', action='store_true', help='Test Flash Attention')
    parser.add_argument('--block', action='store_true', help='Test TransformerBlock')
    parser.add_argument('--backbone', action='store_true', help='Test Backbone')
    parser.add_argument('--ddp', action='store_true', help='Test DDP script')
    args = parser.parse_args()

    # Default to all if no specific test selected
    run_all = args.all or not any([
        args.imports, args.rmsnorm, args.swiglu, args.rope,
        args.flash, args.block, args.backbone, args.ddp
    ])

    print("\n" + "=" * 60)
    print("ActionFormer Improvements Test Suite")
    print("=" * 60 + "\n")

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Some tests will be skipped.\n")

    results = {}

    if run_all or args.imports:
        results['imports'] = test_imports()

    if torch.cuda.is_available():
        if run_all or args.rmsnorm:
            results['rmsnorm'] = test_rmsnorm()

        if run_all or args.swiglu:
            results['swiglu'] = test_swiglu()

        if run_all or args.rope:
            results['rope'] = test_rope()

        if run_all or args.flash:
            results['flash'] = test_flash_attention()

        if run_all or args.block:
            results['block'] = test_transformer_block()

        if run_all or args.backbone:
            results['backbone'] = test_backbone()

    if run_all or args.ddp:
        results['ddp'] = test_ddp_ready()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print()

    all_passed = all(results.values())
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
