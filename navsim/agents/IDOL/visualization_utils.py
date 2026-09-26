from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.common.dataclasses import Scene
from navsim.visualization.bev import (
    add_annotations_to_bev_ax,
    add_lidar_to_bev_ax,
    add_map_to_bev_ax,
)
from navsim.visualization.config import BEV_PLOT_CONFIG


PAPER_COLORS: Dict[str, str] = {
    "expert": "#2f7d32",
    "idol": "#d62728",
    "baseline": "#1f77b4",
    "initial": "#555555",
    "refine1": "#f28e2b",
    "refine2": "#d62728",
    "heat_transition": "magma",
    "heat_idm": "inferno",
    "heat_attention": "turbo",
    "heat_overlap": "viridis",
}


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def infer_bev_hw(num_tokens: int, bev_hw: Optional[Sequence[int]] = None) -> Tuple[int, int]:
    if bev_hw is not None:
        h, w = int(bev_hw[0]), int(bev_hw[1])
        if h * w == int(num_tokens):
            return h, w

    side = int(round(math.sqrt(int(num_tokens))))
    if side * side == int(num_tokens):
        return side, side

    h = int(math.floor(math.sqrt(int(num_tokens))))
    while h > 1 and int(num_tokens) % h != 0:
        h -= 1
    return h, int(num_tokens) // h


def _feature_to_grid(value: Any, bev_hw: Optional[Sequence[int]] = None) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 1:
        h, w = infer_bev_hw(arr.shape[0], bev_hw)
        return arr.reshape(h, w, 1)

    if arr.ndim == 2:
        if bev_hw is not None and tuple(arr.shape) == (int(bev_hw[0]), int(bev_hw[1])):
            return arr[..., None]
        h, w = infer_bev_hw(arr.shape[0], bev_hw)
        return arr.reshape(h, w, arr.shape[-1])

    if arr.ndim == 3:
        if bev_hw is not None:
            h, w = int(bev_hw[0]), int(bev_hw[1])
            if arr.shape[:2] == (h, w):
                return arr
            if arr.shape[-2:] == (h, w):
                return np.moveaxis(arr, 0, -1)
        if arr.shape[0] <= 8 and arr.shape[1] != arr.shape[2]:
            return np.moveaxis(arr, 0, -1)
        return arr

    raise ValueError(f"Cannot convert tensor with shape {arr.shape} to a BEV grid.")


def tensor_to_bev_map(
    bev_tensor: Any,
    mode: str = "mean_abs",
    bev_hw: Optional[Sequence[int]] = None,
) -> np.ndarray:
    grid = _feature_to_grid(bev_tensor, bev_hw=bev_hw)
    if grid.ndim == 2:
        return grid
    if grid.shape[-1] == 1:
        return grid[..., 0]
    if mode == "l2":
        return np.linalg.norm(grid, axis=-1)
    if mode == "max_abs":
        return np.max(np.abs(grid), axis=-1)
    if mode == "mean":
        return np.mean(grid, axis=-1)
    return np.mean(np.abs(grid), axis=-1)


def compute_transition_heatmap(
    bev_states: Any,
    mode: str = "l2",
    bev_hw: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> np.ndarray:
    if isinstance(bev_states, (list, tuple)):
        states = list(bev_states)
    else:
        states = list(_to_numpy(bev_states))

    if len(states) < 2:
        return np.zeros((0, 1, 1), dtype=np.float32)

    grids = [_feature_to_grid(state, bev_hw=bev_hw) for state in states]
    transitions: List[np.ndarray] = []
    for cur_grid, next_grid in zip(grids[:-1], grids[1:]):
        diff = next_grid - cur_grid
        heatmap = tensor_to_bev_map(diff, mode=mode, bev_hw=bev_hw)
        if normalize:
            heatmap = normalize_heatmap(heatmap)
        transitions.append(heatmap.astype(np.float32))
    return np.stack(transitions, axis=0)


def normalize_heatmap(
    heatmap: Any,
    percentile: bool = True,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    arr = _to_numpy(heatmap).astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    valid = arr[finite]
    if percentile:
        lo, hi = np.percentile(valid, [lower, upper])
    else:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    arr[~finite] = 0.0
    return arr


def compute_heatmap_alignment(
    reference: Any,
    response: Any,
    top_percentile: float = 85.0,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compare whether a response heatmap concentrates on the reference hot regions.

    The reference is usually the latent BEV transition between two adjacent
    world-model frames. The response can be an IDM spatial map or IDM attention.
    """
    ref = normalize_heatmap(reference).reshape(-1).astype(np.float32)
    resp = normalize_heatmap(response).reshape(-1).astype(np.float32)
    valid = np.isfinite(ref) & np.isfinite(resp)
    if not valid.any():
        return {
            "pearson": 0.0,
            "cosine": 0.0,
            "topk_iou": 0.0,
            "topk_precision": 0.0,
            "topk_recall": 0.0,
            "response_energy_in_reference_top": 0.0,
        }

    ref = ref[valid]
    resp = resp[valid]
    ref_centered = ref - ref.mean()
    resp_centered = resp - resp.mean()
    pearson = float(
        np.dot(ref_centered, resp_centered)
        / (np.linalg.norm(ref_centered) * np.linalg.norm(resp_centered) + eps)
    )
    cosine = float(np.dot(ref, resp) / (np.linalg.norm(ref) * np.linalg.norm(resp) + eps))

    ref_threshold = np.percentile(ref, top_percentile)
    resp_threshold = np.percentile(resp, top_percentile)
    ref_mask = ref >= ref_threshold
    resp_mask = resp >= resp_threshold
    intersection = float(np.logical_and(ref_mask, resp_mask).sum())
    union = float(np.logical_or(ref_mask, resp_mask).sum())
    ref_count = float(ref_mask.sum())
    resp_count = float(resp_mask.sum())
    resp_energy = float(resp.sum())
    response_energy_in_reference_top = float(resp[ref_mask].sum() / (resp_energy + eps))

    return {
        "pearson": pearson,
        "cosine": cosine,
        "topk_iou": intersection / union if union > 0 else 0.0,
        "topk_precision": intersection / resp_count if resp_count > 0 else 0.0,
        "topk_recall": intersection / ref_count if ref_count > 0 else 0.0,
        "response_energy_in_reference_top": response_energy_in_reference_top,
    }


def heatmap_alignment_score(metrics: Dict[str, float]) -> float:
    return float(
        max(metrics.get("pearson", 0.0), 0.0)
        + metrics.get("topk_recall", 0.0)
        + metrics.get("response_energy_in_reference_top", 0.0)
    )


def configure_paper_bev_ax(ax: plt.Axes, figure_margin: Optional[Tuple[float, float]] = None) -> plt.Axes:
    margin_x, margin_y = figure_margin or BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def add_bev_background(
    ax: plt.Axes,
    scene: Scene,
    frame_idx: Optional[int] = None,
    with_lidar: bool = False,
) -> plt.Axes:
    if frame_idx is None:
        frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*frame.ego_status.ego_pose))
    add_annotations_to_bev_ax(ax, frame.annotations)
    if with_lidar and frame.lidar.lidar_pc is not None:
        add_lidar_to_bev_ax(ax, frame.lidar)
    return ax


def add_trajectory_array_to_ax(
    ax: plt.Axes,
    trajectory: Any,
    label: str,
    color: str,
    linewidth: float = 2.2,
    linestyle: str = "-",
    alpha: float = 1.0,
    marker: Optional[str] = "o",
    zorder: int = 5,
) -> None:
    poses = _to_numpy(trajectory)
    if poses is None or poses.size == 0:
        return
    poses = poses.reshape(-1, poses.shape[-1])
    xy = np.concatenate([np.array([[0.0, 0.0]], dtype=np.float32), poses[:, :2]], axis=0)
    ax.plot(
        xy[:, 1],
        xy[:, 0],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        marker=marker,
        markersize=3.8,
        markeredgewidth=0.4,
        markeredgecolor="white",
        label=label,
        zorder=zorder,
    )


def plot_heatmap_overlay(
    ax: plt.Axes,
    heatmap: Any,
    scene: Optional[Scene] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
    alpha: float = 0.62,
    extent: Tuple[float, float, float, float] = (-32.0, 32.0, 0.0, 32.0),
    with_lidar: bool = False,
    contour_heatmap: Optional[Any] = None,
    contour_color: str = "white",
    contour_label: Optional[str] = None,
) -> plt.Axes:
    if scene is not None:
        add_bev_background(ax, scene, with_lidar=with_lidar)
    heatmap = normalize_heatmap(heatmap)
    ax.imshow(
        heatmap,
        cmap=cmap,
        origin="lower",
        extent=extent,
        alpha=alpha,
        interpolation="bilinear",
        zorder=4,
    )
    if contour_heatmap is not None:
        contour = normalize_heatmap(contour_heatmap)
        max_contour = float(np.nanmax(contour))
        if max_contour > 0:
            level = float(np.percentile(contour, 85.0))
            if level <= 0.0 or level >= max_contour:
                level = 0.5 * max_contour
            ax.contour(
                contour,
                levels=[level],
                colors=contour_color,
                linewidths=1.2,
                origin="lower",
                extent=extent,
                zorder=8,
            )
            if contour_label:
                ax.text(
                    0.98,
                    0.04,
                    contour_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color=contour_color,
                    bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5),
                    zorder=20,
                )
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    configure_paper_bev_ax(ax)
    return ax


def save_figure(fig: plt.Figure, output_path: Path, save_pdf: bool = True, dpi: int = 220) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_figure3_qualitative(samples: Sequence[Dict[str, Any]]) -> plt.Figure:
    if len(samples) == 0:
        raise ValueError("make_figure3_qualitative requires at least one sample.")

    num_panels = min(4, len(samples))
    cols = 2 if num_panels > 1 else 1
    rows = int(math.ceil(num_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes_arr.ravel()[num_panels:]:
        ax.axis("off")

    for idx, sample in enumerate(samples[:num_panels]):
        ax = axes_arr.ravel()[idx]
        add_bev_background(ax, sample["scene"], with_lidar=sample.get("with_lidar", False))
        if sample.get("baseline_trajectory") is not None:
            add_trajectory_array_to_ax(
                ax,
                sample["baseline_trajectory"],
                "Baseline",
                PAPER_COLORS["baseline"],
                linewidth=1.8,
                linestyle="--",
                alpha=0.9,
                zorder=5,
            )
        add_trajectory_array_to_ax(
            ax,
            sample["gt_trajectory"],
            "Expert",
            PAPER_COLORS["expert"],
            linewidth=2.0,
            linestyle="-",
            alpha=0.95,
            zorder=6,
        )
        add_trajectory_array_to_ax(
            ax,
            sample["idol_trajectory"],
            "IDOL",
            PAPER_COLORS["idol"],
            linewidth=2.5,
            linestyle="-",
            alpha=1.0,
            zorder=7,
        )
        token = sample.get("token", "")
        ax.set_title(f"Scene {idx + 1}: {token[:8]}", fontsize=10, pad=4)
        configure_paper_bev_ax(ax)
        ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.86)

    fig.suptitle("Qualitative trajectory results", fontsize=13)
    fig.tight_layout()
    return fig


def make_single_sample_trajectory_figure(sample: Dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
    add_bev_background(ax, sample["scene"], with_lidar=sample.get("with_lidar", False))
    if sample.get("baseline_trajectory") is not None:
        add_trajectory_array_to_ax(
            ax,
            sample["baseline_trajectory"],
            "Baseline",
            PAPER_COLORS["baseline"],
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
        )
    add_trajectory_array_to_ax(ax, sample["gt_trajectory"], "Expert", PAPER_COLORS["expert"])
    add_trajectory_array_to_ax(
        ax,
        sample["idol_trajectory"],
        "IDOL",
        PAPER_COLORS["idol"],
        linewidth=2.6,
    )
    ax.set_title("Qualitative trajectory result", fontsize=10, pad=4)
    configure_paper_bev_ax(ax)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.86)
    fig.tight_layout()
    return fig


def make_figure4_idm_refinement(sample: Dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(18.8, 4.6))

    transition_maps = _to_numpy(sample.get("future_transition_maps"))
    if transition_maps is None or transition_maps.size == 0:
        transition_step = np.zeros(tuple(sample.get("bev_hw", (8, 8))), dtype=np.float32)
    else:
        step_idx = int(sample.get("idm_evidence_step_idx", 0))
        step_idx = max(0, min(step_idx, transition_maps.shape[0] - 1))
        transition_step = normalize_heatmap(transition_maps[step_idx])

    idm_heatmaps = _to_numpy(sample.get("idm_spatial_heatmaps"))
    if idm_heatmaps is not None and idm_heatmaps.size > 0:
        step_idx = int(sample.get("idm_evidence_step_idx", 0))
        step_idx = max(0, min(step_idx, idm_heatmaps.shape[0] - 1))
        idm_map = normalize_heatmap(idm_heatmaps[step_idx])
    else:
        idm_map = sample.get("idm_spatial_map_avg")
    if idm_map is None:
        idm_map = np.zeros_like(transition_step)

    attention_map = sample.get("idm_attention_map_final")
    if attention_map is None:
        attention_map = idm_map
    attention_map = normalize_heatmap(attention_map)

    plot_heatmap_overlay(
        axes[0],
        transition_step,
        scene=sample["scene"],
        title="Latent BEV change: frame t -> t+1",
        cmap=PAPER_COLORS["heat_transition"],
        alpha=0.64,
    )
    plot_heatmap_overlay(
        axes[1],
        idm_map,
        scene=sample["scene"],
        title="IDM dynamics from BEV pair",
        cmap=PAPER_COLORS["heat_idm"],
        alpha=0.64,
        contour_heatmap=transition_step,
    )
    plot_heatmap_overlay(
        axes[2],
        attention_map,
        scene=sample["scene"],
        title="IDM attention on dynamic queries",
        cmap=PAPER_COLORS["heat_attention"],
        alpha=0.66,
        contour_heatmap=transition_step,
    )

    ax = axes[3]
    add_bev_background(ax, sample["scene"], with_lidar=sample.get("with_lidar", False))
    refinement_trajs = list(sample.get("refinement_trajectories", []))
    trajectory_panels = []
    if refinement_trajs:
        trajectory_panels.append(("Original", refinement_trajs[0], PAPER_COLORS["initial"], "--", 1.8, 5))
        trajectory_panels.append(("Refined", refinement_trajs[-1], PAPER_COLORS["refine2"], "-", 2.6, 7))

    for label, trajectory, color, linestyle, linewidth, zorder in trajectory_panels:
        add_trajectory_array_to_ax(
            ax,
            trajectory,
            label,
            color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.95,
            zorder=zorder,
        )
    add_trajectory_array_to_ax(
        ax,
        sample["gt_trajectory"],
        "Expert",
        PAPER_COLORS["expert"],
        linewidth=2.0,
        linestyle="-",
        alpha=0.85,
        zorder=8,
    )
    ax.set_title("Query refinement trajectory", fontsize=10, pad=4)
    configure_paper_bev_ax(ax)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.86)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
            zorder=20,
        )

    fig.tight_layout()
    return fig


def add_front_camera_to_ax(ax: plt.Axes, sample: Dict[str, Any]) -> None:
    scene = sample["scene"]
    frame_idx = scene.scene_metadata.num_history_frames - 1
    image = scene.frames[frame_idx].cameras.cam_f0.image
    if image is None:
        ax.axis("off")
        ax.set_title("Front camera unavailable", fontsize=10, pad=4)
        return
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Front camera", fontsize=10, pad=4)


def add_failure_trajectory_panel(ax: plt.Axes, sample: Dict[str, Any]) -> None:
    add_bev_background(ax, sample["scene"], with_lidar=sample.get("with_lidar", False))
    refinement_trajs = list(sample.get("refinement_trajectories", []))
    if refinement_trajs:
        add_trajectory_array_to_ax(
            ax,
            refinement_trajs[0],
            "Original",
            PAPER_COLORS["initial"],
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
            zorder=5,
        )
        add_trajectory_array_to_ax(
            ax,
            refinement_trajs[-1],
            "Refined",
            PAPER_COLORS["refine2"],
            linewidth=2.6,
            linestyle="-",
            alpha=0.95,
            zorder=7,
        )
    add_trajectory_array_to_ax(
        ax,
        sample["gt_trajectory"],
        "Expert",
        PAPER_COLORS["expert"],
        linewidth=2.0,
        linestyle="-",
        alpha=0.9,
        zorder=8,
    )
    ax.set_title("Trajectory comparison", fontsize=10, pad=4)
    configure_paper_bev_ax(ax)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.86)


def make_failure_cases_figure(samples: Sequence[Dict[str, Any]], max_cases: int = 2) -> plt.Figure:
    if len(samples) == 0:
        raise ValueError("make_failure_cases_figure requires at least one sample.")

    num_cases = min(max_cases, len(samples))
    fig, axes = plt.subplots(num_cases, 2, figsize=(10.8, 4.2 * num_cases))
    axes_arr = np.asarray(axes).reshape(num_cases, 2)

    for idx, sample in enumerate(samples[:num_cases]):
        add_front_camera_to_ax(axes_arr[idx, 0], sample)
        add_failure_trajectory_panel(axes_arr[idx, 1], sample)
        case_label = f"({chr(ord('a') + idx)})"
        for ax in axes_arr[idx]:
            ax.text(
                0.02,
                0.98,
                case_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                zorder=20,
            )

    fig.tight_layout()
    return fig
