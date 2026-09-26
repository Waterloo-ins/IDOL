"""Fixed motion features for supervising IDM, independent of the trajectory decoder."""

import torch
import torch.nn.functional as F


def motion_features(trajectory, dt, scales, min_speed):
    """Return [v(1:T), a(1:T-1), yaw_rate(1:T)] and validity masks.

    Poses are in the current ego frame. Speed is the signed displacement
    projected onto the midpoint heading; acceleration is between interval
    speeds, not a difference against the instantaneous ego speed.
    """
    if dt <= 0 or len(scales) != 3 or any(s <= 0 for s in scales):
        raise ValueError("Motion dt and the three feature scales must be positive")
    if min_speed < 0:
        raise ValueError("Motion min_speed must be nonnegative")
    trajectory = trajectory.float()
    valid = torch.isfinite(trajectory).all(dim=-1)
    trajectory = torch.where(torch.isfinite(trajectory), trajectory, 0.0)
    poses = torch.cat([torch.zeros_like(trajectory[..., :1, :]), trajectory], dim=-2)
    valid = torch.cat([torch.ones_like(valid[..., :1]), valid], dim=-1)
    segment_valid = valid[..., 1:] & valid[..., :-1]
    delta = poses[..., 1:, :] - poses[..., :-1, :]
    angle = torch.atan2(delta[..., 2].sin(), delta[..., 2].cos())
    midpoint = poses[..., :-1, 2] + 0.5 * angle
    speed = (delta[..., 0] * midpoint.cos() + delta[..., 1] * midpoint.sin()) / dt
    acceleration = (speed[..., 1:] - speed[..., :-1]) / dt
    yaw_rate = angle / dt
    # Heading at standstill is not a reliable angular-velocity label.
    yaw_valid = segment_valid & (speed.abs() >= min_speed)
    features = torch.cat([
        speed / scales[0], acceleration / scales[1], yaw_rate / scales[2]
    ], dim=-1)
    mask = torch.cat([
        segment_valid, segment_valid[..., 1:] & segment_valid[..., :-1], yaw_valid
    ], dim=-1)
    return features, mask


def compute_idm_latent_corr_loss(predictions, targets, config):
    """WTA supervision of the final IDM readout; target construction has no gradient."""
    prediction = predictions["idm_motion_residual"]
    with torch.no_grad():
        gt = targets["trajectory"].squeeze(1).float()
        anchors = predictions["trajectory_anchors"].float()
        if not torch.isfinite(anchors).all():
            raise ValueError("Trajectory anchors must be finite")
        # Match the existing offset loss assignment, including heading distance.
        valid_sample = torch.isfinite(gt).all(dim=(-2, -1))
        safe_gt = torch.where(torch.isfinite(gt), gt, 0.0)
        distance = torch.linalg.vector_norm(
            anchors.flatten(1)[None] - safe_gt.flatten(1)[:, None], dim=-1
        )
        winner = distance.argmin(dim=-1)
        args = (config.trajectory_sampling.interval_length,
                config.idm_motion_scales, config.idm_motion_min_speed)
        gt_feature, gt_mask = motion_features(gt, *args)
        anchor_feature, anchor_mask = motion_features(anchors[winner], *args)
        residual = gt_feature - anchor_feature
        mask = gt_mask & anchor_mask & valid_sample[:, None]

    selected = prediction[torch.arange(gt.shape[0], device=prediction.device), winner].float()
    if selected.shape != residual.shape:
        raise ValueError("IDM motion readout and target feature dimensions do not match")
    error = F.smooth_l1_loss(selected, residual, reduction="none")
    # Average each physical component separately so sequence length does not
    # silently change the relative weight of acceleration and angular velocity.
    count = gt.shape[-2]
    sections = ((0, count), (count, 2 * count - 1), (2 * count - 1, 3 * count - 1))
    losses = []
    for start, end in sections:
        weights = mask[..., start:end].to(error.dtype)
        losses.append((error[..., start:end] * weights).sum() / weights.sum().clamp_min(1))
    return sum(losses) / len(losses)
