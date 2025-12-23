# Functions for 1D NMS, modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import torch
from typing import Dict, Optional, Union

import nms_1d_cpu


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, min_score, max_num
    ):
        # vanilla nms will not change the score, so we can filter segs first
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        # nms op; return inds that is sorted by descending order
        inds = nms_1d_cpu.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_threshold=float(iou_threshold))
        # cap by max number
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        # return the sorted segs / scores
        sorted_segs = segs[inds]
        sorted_scores = scores[inds]
        sorted_cls_idxs = cls_idxs[inds]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, sigma, min_score, method, max_num
    ):
        # pre allocate memory for sorted results
        dets = segs.new_empty((segs.size(0), 3), device='cpu')
        # softnms op, return dets that stores the sorted segs / scores
        inds = nms_1d_cpu.softnms(
            segs.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method))
        # cap by max number
        if max_num > 0:
            n_segs = min(len(inds), max_num)
        else:
            n_segs = len(inds)
        sorted_segs = dets[:n_segs, :2]
        sorted_scores = dets[:n_segs, 2]
        sorted_cls_idxs = cls_idxs[inds]
        sorted_cls_idxs = sorted_cls_idxs[:n_segs]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """
        blur localization results by incorporating side segs.
        this is known as bounding box voting in object detection literature.
        slightly boost the performance around iou_threshold
    """

    # *_segs : N_i x 2, all_scores: N,
    # apply offset
    offset_scores = all_scores + score_offset

    # computer overlap between nms and all segs
    # construct the distance matrix of # N_nms x # N_all
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    # compute intersection
    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right-left).clamp(min=0)

    # lens of all segments
    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]

    # iou
    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    # get neighbors (# N_nms x # N_all) / weights
    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :] * iou
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    refined_segs = seg_weights @ all_segs

    return refined_segs


def compute_diou_1d(segs1: torch.Tensor, segs2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute DIoU (Distance-IoU) matrix for 1D segments.

    Args:
        segs1: (N, 2) tensor of segments [start, end]
        segs2: (M, 2) tensor of segments [start, end]

    Returns:
        (N, M) DIoU matrix
    """
    N, M = segs1.shape[0], segs2.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=segs1.dtype, device=segs1.device)

    # Expand for broadcasting: (N, 1, 2) and (1, M, 2)
    segs1_exp = segs1[:, None, :]
    segs2_exp = segs2[None, :, :]

    # Intersection
    left = torch.maximum(segs1_exp[:, :, 0], segs2_exp[:, :, 0])
    right = torch.minimum(segs1_exp[:, :, 1], segs2_exp[:, :, 1])
    inter = (right - left).clamp(min=0)

    # Union
    len1 = segs1_exp[:, :, 1] - segs1_exp[:, :, 0]
    len2 = segs2_exp[:, :, 1] - segs2_exp[:, :, 0]
    union = len1 + len2 - inter

    # IoU
    iou = inter / union.clamp(min=eps)

    # Center distance
    center1 = (segs1_exp[:, :, 0] + segs1_exp[:, :, 1]) / 2
    center2 = (segs2_exp[:, :, 0] + segs2_exp[:, :, 1]) / 2
    center_dist_sq = (center1 - center2) ** 2

    # Enclosing segment
    enc_left = torch.minimum(segs1_exp[:, :, 0], segs2_exp[:, :, 0])
    enc_right = torch.maximum(segs1_exp[:, :, 1], segs2_exp[:, :, 1])
    enc_diag_sq = (enc_right - enc_left) ** 2

    # DIoU = IoU - d²/c²
    diou = iou - center_dist_sq / enc_diag_sq.clamp(min=eps)

    return diou


def diou_soft_nms(
    segs: torch.Tensor,
    scores: torch.Tensor,
    cls_idxs: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    min_score: float = 0.001,
    max_num: int = -1,
) -> tuple:
    """
    DIoU-based Soft-NMS implementation.
    Uses DIoU instead of IoU for suppression, considering center distance.

    Args:
        segs: (N, 2) segments [start, end]
        scores: (N,) confidence scores
        cls_idxs: (N,) class indices
        iou_threshold: IoU threshold (not used in soft-nms, kept for API compat)
        sigma: Gaussian sigma for score decay
        min_score: Minimum score threshold
        max_num: Maximum detections to keep

    Returns:
        Tuple of (segs, scores, cls_idxs) after NMS
    """
    device = segs.device
    segs = segs.cpu().clone()
    scores = scores.cpu().clone()
    cls_idxs = cls_idxs.cpu().clone()

    N = segs.shape[0]
    if N == 0:
        return segs.to(device), scores.to(device), cls_idxs.to(device)

    order = scores.argsort(descending=True)
    segs = segs[order]
    scores = scores[order]
    cls_idxs = cls_idxs[order]

    kept_segs = []
    kept_scores = []
    kept_cls = []

    while segs.shape[0] > 0:
        kept_segs.append(segs[0])
        kept_scores.append(scores[0])
        kept_cls.append(cls_idxs[0])

        if segs.shape[0] == 1:
            break

        top_seg = segs[0:1]
        rest_segs = segs[1:]
        diou = compute_diou_1d(top_seg, rest_segs).squeeze(0)

        diou_clamped = diou.clamp(min=0)
        weight = torch.exp(-(diou_clamped ** 2) / sigma)
        scores = scores[1:] * weight

        valid = scores > min_score
        segs = rest_segs[valid]
        scores = scores[valid]
        cls_idxs = cls_idxs[1:][valid]

    if len(kept_segs) == 0:
        return (
            torch.zeros((0, 2), device=device),
            torch.zeros(0, device=device),
            torch.zeros(0, dtype=cls_idxs.dtype, device=device)
        )

    kept_segs = torch.stack(kept_segs)
    kept_scores = torch.stack(kept_scores)
    kept_cls = torch.stack(kept_cls)

    if max_num > 0:
        kept_segs = kept_segs[:max_num]
        kept_scores = kept_scores[:max_num]
        kept_cls = kept_cls[:max_num]

    return kept_segs.to(device), kept_scores.to(device), kept_cls.to(device)


def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold,
    min_score,
    max_seg_num,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.75,
    use_diou_nms=False,
    class_sigma: Optional[Dict[int, float]] = None,
):
    num_segs = segs.shape[0]
    if num_segs == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    if multiclass:
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            curr_sigma = sigma
            if class_sigma is not None and int(class_id.item()) in class_sigma:
                curr_sigma = class_sigma[int(class_id.item())]

            if use_diou_nms and use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = diou_soft_nms(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    curr_sigma,
                    min_score,
                    max_seg_num
                )
            elif use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    curr_sigma,
                    min_score,
                    2,
                    max_seg_num
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs = NMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    min_score,
                    max_seg_num
                )

            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)

    else:
        if use_diou_nms and use_soft_nms:
            new_segs, new_scores, new_cls_idxs = diou_soft_nms(
                segs, scores, cls_idxs, iou_threshold,
                sigma, min_score, max_seg_num
            )
        elif use_soft_nms:
            new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                sigma, min_score, 2, max_seg_num
            )
        else:
            new_segs, new_scores, new_cls_idxs = NMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                min_score, max_seg_num
            )
        # seg voting
        if voting_thresh > 0:
            new_segs = seg_voting(
                new_segs,
                segs,
                scores,
                voting_thresh
            )

    # sort based on scores and return
    # truncate the results based on max_seg_num
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    # needed for multiclass NMS
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
