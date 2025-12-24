#!/usr/bin/env python3
"""
Evaluate Snap Detection V7b Model

Uses ActionFormer's built-in inference which includes:
- DIoU-NMS (from test_cfg.use_diou_nms)
- Soft NMS (from test_cfg.nms_method)
- Score temperature scaling
- EMA model weights

Coordinate conversion:
- Predictions are in feature frame space
- Convert: seconds = feature_frame / 14.5
- Convert: video_frame = feature_frame / 14.5 * 59
"""

import os
import sys
import json
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from pprint import pprint

sys.path.insert(0, '/home/ubuntu/ActionFormer')

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection, fix_random_seed, valid_one_epoch
from libs.utils.temporal_constraints import TemporalConstraints

# Coordinate conversion constants
FEATURES_PER_SEC = 14.5  # ~59 fps / 4 stride
VIDEO_FPS = 59.0


def feature_to_seconds(feat_frame):
    """Convert feature frame to seconds."""
    return feat_frame / FEATURES_PER_SEC


def run_standard_eval(config_path, ckpt_path, devices=[0]):
    """Run standard ActionFormer evaluation with all enhancements."""

    # Load config
    cfg = load_config(config_path)

    print("\n" + "=" * 60)
    print("EVALUATION CONFIG")
    print("=" * 60)
    print(f"Backbone: {cfg['model']['backbone_type']}")
    print(f"use_abs_pe: {cfg['model'].get('use_abs_pe', False)}")
    print(f"use_rope: {cfg['model'].get('use_rope', False)}")
    print(f"DIoU-NMS: {cfg['test_cfg'].get('use_diou_nms', False)}")
    print(f"NMS method: {cfg['test_cfg'].get('nms_method', 'hard')}")
    print(f"Score temp: {cfg['test_cfg'].get('score_temperature', 1.0)}")

    # Fix random seed
    fix_random_seed(0, include_cuda=True)

    # Create validation dataset
    val_dataset = make_dataset(
        cfg['dataset_name'],
        False,  # is_training
        cfg['val_split'],
        **cfg['dataset']
    )

    val_loader = make_data_loader(
        val_dataset,
        False,  # is_training
        None,   # generator
        1,      # batch_size
        cfg['loader']['num_workers']
    )

    print(f"\nValidation set: {len(val_dataset)} videos")

    # Create model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=devices)

    # Load checkpoint
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=f'cuda:{devices[0]}')

    # Try EMA weights first, fallback to regular
    if 'state_dict_ema' in checkpoint:
        print("Using EMA model weights")
        model.load_state_dict(checkpoint['state_dict_ema'])
    else:
        print("Using regular model weights")
        model.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint

    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)

    # Use valid_one_epoch for standard evaluation
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds=val_db_vars['tiou_thresholds']
    )

    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,  # curr_epoch
        evaluator=det_eval,
        output_file=None,
        ext_score_file=cfg['test_cfg'].get('ext_score_file'),
        tb_writer=None,
        print_freq=50
    )

    return mAP, val_dataset, cfg


def compute_snap_metrics(val_dataset, predictions_file=None):
    """Compute snap-specific metrics (MAE, Recall@threshold)."""

    # This would require capturing predictions from valid_one_epoch
    # For now, we rely on the mAP from ANET evaluation
    pass


def main():
    parser = argparse.ArgumentParser(description='Evaluate Snap V7b')
    parser.add_argument('--config', default='configs/snap_v7b.yaml')
    parser.add_argument('--ckpt', default=None, help='Checkpoint path (auto-detect if not specified)')
    parser.add_argument('--epoch', type=int, default=-1, help='Specific epoch (-1 = latest)')
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    # Find checkpoint
    if args.ckpt is None:
        ckpt_dir = Path('/home/ubuntu/ActionFormer/ckpt/snap_v7b')
        subdirs = list(ckpt_dir.glob('snap_v7b_ddp_*'))
        if not subdirs:
            print("No checkpoint directory found!")
            return

        latest_dir = sorted(subdirs)[-1]

        if args.epoch > 0:
            ckpt_path = latest_dir / f'epoch_{args.epoch:03d}.pth.tar'
        else:
            ckpts = sorted(latest_dir.glob('epoch_*.pth.tar'))
            if not ckpts:
                print("No checkpoint files found!")
                return
            ckpt_path = ckpts[-1]

        args.ckpt = str(ckpt_path)

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Devices: {args.devices}")

    # Run evaluation
    mAP, val_dataset, cfg = run_standard_eval(args.config, args.ckpt, args.devices)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"mAP: {mAP:.4f}")
    print("\nNote: Predictions are in feature frames.")
    print(f"Convert to seconds: sec = feat_frame / {FEATURES_PER_SEC}")
    print(f"Convert to video frame: vf = feat_frame / {FEATURES_PER_SEC} * {VIDEO_FPS}")


if __name__ == '__main__':
    main()
