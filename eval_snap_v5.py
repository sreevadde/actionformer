#!/usr/bin/env python3
"""
Snap Detection V5 Evaluation Script

Evaluates the V5 ActionFormer model with temporal constraints
and computes snap-specific metrics (MAE, recall at thresholds).
"""

import os
import json
import argparse
import numpy as np
from pprint import pprint

import torch
from torch.utils.data import DataLoader

# ActionFormer imports
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from libs.utils.temporal_constraints import TemporalConstraints, enforce_class_constraints


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Snap V5 Model')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--split', type=str, default='validation', help='Split to evaluate')
    parser.add_argument('--min-gap', type=float, default=3.0,
                        help='Minimum gap between snap detections (seconds)')
    parser.add_argument('--score-thresh', type=float, default=0.3,
                        help='Minimum score threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for predictions')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def load_ground_truth(json_file, split='validation'):
    """Load ground truth snap times from annotations."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    gt = {}
    for video_id, info in data['database'].items():
        if info['subset'] != split:
            continue

        # Get snap annotations (class 0)
        snap_times = []
        for ann in info.get('annotations', []):
            if ann['label_id'] == 0:  # snap class
                # Use center of segment as snap time
                snap_time = (ann['segment'][0] + ann['segment'][1]) / 2
                snap_times.append(snap_time)

        if snap_times:
            gt[video_id] = sorted(snap_times)

    return gt


def run_inference(model, data_loader, device):
    """Run inference on dataset."""
    model.eval()
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': [],
    }

    with torch.no_grad():
        for video_list in data_loader:
            # Forward pass - model expects list of video dicts
            output = model(video_list)

            # Collect results
            for vid_idx, vid_results in enumerate(output):
                video_id = video_list[vid_idx]['video_id']

                if vid_results['segments'].shape[0] == 0:
                    continue

                segments = vid_results['segments'].cpu().numpy()
                scores = vid_results['scores'].cpu().numpy()
                labels = vid_results['labels'].cpu().numpy()

                for i in range(len(segments)):
                    results['video-id'].append(video_id)
                    results['t-start'].append(segments[i, 0])
                    results['t-end'].append(segments[i, 1])
                    results['label'].append(labels[i])
                    results['score'].append(scores[i])

    # Convert to numpy arrays
    results['t-start'] = np.array(results['t-start'])
    results['t-end'] = np.array(results['t-end'])
    results['label'] = np.array(results['label'])
    results['score'] = np.array(results['score'])

    return results


def compute_snap_metrics(predictions, ground_truth, thresholds_ms=[100, 200, 500, 1000]):
    """
    Compute snap detection metrics.

    Args:
        predictions: Dict with video-id, t-start, t-end, score
        ground_truth: Dict mapping video_id -> list of snap times
        thresholds_ms: List of thresholds in milliseconds

    Returns:
        Dict with metrics
    """
    # Group predictions by video
    pred_by_video = {}
    for i, vid in enumerate(predictions['video-id']):
        if vid not in pred_by_video:
            pred_by_video[vid] = []

        pred_time = (predictions['t-start'][i] + predictions['t-end'][i]) / 2
        pred_score = predictions['score'][i]
        pred_by_video[vid].append((pred_time, pred_score))

    # Sort by score (descending) within each video
    for vid in pred_by_video:
        pred_by_video[vid] = sorted(pred_by_video[vid], key=lambda x: -x[1])

    # Compute metrics
    all_errors = []
    recall_counts = {t: 0 for t in thresholds_ms}
    total_gt = 0
    matched_gt = 0
    false_positives = 0

    for vid, gt_times in ground_truth.items():
        total_gt += len(gt_times)

        preds = pred_by_video.get(vid, [])

        # Match predictions to ground truth (greedy, by score)
        gt_matched = [False] * len(gt_times)

        for pred_time, pred_score in preds:
            # Find closest unmatched GT
            best_idx = None
            best_error = float('inf')

            for i, gt_time in enumerate(gt_times):
                if gt_matched[i]:
                    continue
                error = abs(pred_time - gt_time)
                if error < best_error:
                    best_error = error
                    best_idx = i

            if best_idx is not None and best_error < 5.0:  # 5 second max match distance
                gt_matched[best_idx] = True
                matched_gt += 1
                error_ms = best_error * 1000
                all_errors.append(error_ms)

                for t in thresholds_ms:
                    if error_ms <= t:
                        recall_counts[t] += 1
            else:
                false_positives += 1

    # Compute final metrics
    metrics = {
        'total_gt': total_gt,
        'total_pred': len(predictions['video-id']),
        'matched': matched_gt,
        'false_positives': false_positives,
    }

    if all_errors:
        metrics['mae_ms'] = np.mean(all_errors)
        metrics['median_error_ms'] = np.median(all_errors)
        metrics['std_error_ms'] = np.std(all_errors)
    else:
        metrics['mae_ms'] = float('inf')
        metrics['median_error_ms'] = float('inf')
        metrics['std_error_ms'] = 0

    for t in thresholds_ms:
        recall = recall_counts[t] / total_gt * 100 if total_gt > 0 else 0
        metrics[f'recall@{t}ms'] = recall

    if matched_gt + false_positives > 0:
        metrics['precision'] = matched_gt / (matched_gt + false_positives) * 100
    else:
        metrics['precision'] = 0

    return metrics


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Fix random seed
    fix_random_seed(cfg['init_rand_seed'])

    print("=" * 60)
    print("Snap V5 Evaluation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Split: {args.split}")
    print(f"Min gap: {args.min_gap}s")
    print(f"Score threshold: {args.score_thresh}")
    print("=" * 60)

    # Build model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = model.to(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)

    # Handle DDP state dict
    state_dict = ckpt['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")

    # Build dataset
    split_name = cfg['val_split'] if args.split == 'validation' else cfg['train_split']
    val_dataset = make_dataset(
        cfg['dataset_name'],
        False,  # is_training=False for evaluation
        split_name,
        **cfg['dataset']
    )

    val_loader = make_data_loader(
        val_dataset,
        is_training=False,
        generator=None,
        batch_size=1,
        num_workers=2
    )

    print(f"Validation set: {len(val_dataset)} videos")

    # Run inference
    print("\nRunning inference...")
    results = run_inference(model, val_loader, args.device)
    print(f"Raw detections: {len(results['video-id'])}")

    # Apply temporal constraints
    print(f"\nApplying temporal constraints (min_gap={args.min_gap}s, score>{args.score_thresh})...")
    constraints = TemporalConstraints(
        min_gap={0: args.min_gap},  # Class 0 = snap
        max_overlap=0.3,
        score_threshold=args.score_thresh,
    )
    filtered_results = constraints.apply(results)
    print(f"After constraints: {len(filtered_results['video-id'])} detections")

    # Load ground truth
    gt = load_ground_truth(cfg['dataset']['json_file'], args.split)
    print(f"Ground truth: {sum(len(v) for v in gt.values())} snaps in {len(gt)} videos")

    # Compute metrics
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    metrics = compute_snap_metrics(filtered_results, gt)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Save predictions if requested
    if args.output:
        output_data = {
            'config': args.config,
            'checkpoint': args.ckpt,
            'metrics': metrics,
            'predictions': {
                'video-id': filtered_results['video-id'],
                't-start': filtered_results['t-start'].tolist(),
                't-end': filtered_results['t-end'].tolist(),
                'label': filtered_results['label'].tolist(),
                'score': filtered_results['score'].tolist(),
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved predictions to {args.output}")


if __name__ == '__main__':
    main()
