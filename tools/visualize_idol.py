from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/navsim_matplotlib_{os.environ.get('USER', 'user')}")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.IDOL.IDOL_agent import IDOLAgent
from navsim.agents.IDOL.configs.default import IDOLConfig
from navsim.agents.IDOL.visualization_utils import (
    compute_heatmap_alignment,
    compute_transition_heatmap,
    heatmap_alignment_score,
    make_failure_cases_figure,
    make_figure3_qualitative,
    make_figure4_idm_refinement,
    make_single_sample_trajectory_figure,
    normalize_heatmap,
    save_figure,
    tensor_to_bev_map,
)
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.enums import BoundingBoxIndex


LOGGER = logging.getLogger("visualize_idol")


def _parse_tokens(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    tokens: List[str] = []
    for value in values:
        tokens.extend([item.strip() for item in value.split(",") if item.strip()])
    return tokens


def _sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:96]


def _resolve_checkpoint_path(checkpoint_path: str, arg_name: str) -> str:
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{arg_name} not found: {checkpoint}. "
            "Replace the example '/path/to/checkpoint.ckpt' with a real checkpoint path."
        )
    return str(checkpoint)


def _resolve_split(split: str, data_split: Optional[str], scene_filter: Optional[str]) -> Tuple[str, str]:
    split_alias = {
        "navtest": ("test", "navtest"),
        "test": ("test", "navtest"),
        "mini": ("trainval", "navtrain_mini"),
        "navtrain_mini": ("trainval", "navtrain_mini"),
        "train": ("trainval", "navtrain"),
        "trainval": ("trainval", "all_scenes"),
        "val": ("trainval", "all_scenes"),
    }
    resolved_data_split, resolved_scene_filter = split_alias.get(split, (split, "all_scenes"))
    return data_split or resolved_data_split, scene_filter or resolved_scene_filter


def _configure_matplotlib_font(font_family: Optional[str]) -> None:
    if not font_family:
        return

    if font_family.lower() == "times new roman":
        for font_path in (
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf",
        ):
            if Path(font_path).is_file():
                font_manager.fontManager.addfont(font_path)

    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.serif": [font_family],
            "font.sans-serif": [font_family],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_scene_filter(
    scene_filter_name: str,
    sample_tokens: Sequence[str],
    num_samples: int,
    log_names: Sequence[str],
) -> SceneFilter:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "common"
        / "scene_filter"
        / f"{scene_filter_name}.yaml"
    )
    if config_path.is_file():
        scene_filter = instantiate(OmegaConf.load(config_path))
    else:
        LOGGER.warning("Scene filter config %s not found. Falling back to all_scenes.", config_path)
        scene_filter = SceneFilter()

    if sample_tokens:
        scene_filter.tokens = list(sample_tokens)
        scene_filter.max_scenes = None
    elif num_samples > 0:
        scene_filter.max_scenes = max(num_samples, 1)

    if log_names:
        scene_filter.log_names = list(log_names)

    return scene_filter


def _apply_config_overrides(config: IDOLConfig, overrides: Sequence[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Config override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        value: Any = raw_value.strip()
        lower = value.lower() if isinstance(value, str) else value
        if lower == "true":
            value = True
        elif lower == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = raw_value
        setattr(config, key, value)


def _build_agent(
    checkpoint_path: str,
    device: torch.device,
    config_overrides: Sequence[str],
) -> IDOLAgent:
    config = IDOLConfig()
    _apply_config_overrides(config, config_overrides)
    agent = IDOLAgent(
        config=config,
        trajectory_sampling=config.trajectory_sampling,
        lr=2e-4,
        checkpoint_path=checkpoint_path,
    )
    agent.initialize()
    agent.to(device)
    agent.eval()
    agent.is_eval = True
    return agent


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _select_candidate(value: Any, selected_idx: int, batch_idx: int = 0) -> np.ndarray:
    arr = _tensor_to_numpy(value)
    return arr[batch_idx, selected_idx]


def _select_evidence_step(
    transition_maps: np.ndarray,
    idm_spatial_heatmaps: np.ndarray,
) -> Tuple[int, List[Dict[str, float]], Dict[str, float]]:
    if transition_maps.size == 0 or idm_spatial_heatmaps.size == 0:
        return 0, [], {}

    num_steps = min(len(transition_maps), len(idm_spatial_heatmaps))
    alignments: List[Dict[str, float]] = []
    for step_idx in range(num_steps):
        metrics = compute_heatmap_alignment(
            transition_maps[step_idx],
            idm_spatial_heatmaps[step_idx],
        )
        metrics["transition_idx"] = float(step_idx)
        metrics["score"] = heatmap_alignment_score(metrics)
        alignments.append(metrics)

    best_idx = max(range(num_steps), key=lambda idx: alignments[idx]["score"])
    return best_idx, alignments, alignments[best_idx]


def _trajectory_error_metrics(initial: np.ndarray, refined: np.ndarray, expert: np.ndarray) -> Dict[str, float]:
    initial = np.asarray(initial, dtype=np.float32)
    refined = np.asarray(refined, dtype=np.float32)
    expert = np.asarray(expert, dtype=np.float32)
    num_steps = min(len(initial), len(refined), len(expert))
    if num_steps == 0:
        return {
            "initial_ade": 0.0,
            "refined_ade": 0.0,
            "initial_fde": 0.0,
            "refined_fde": 0.0,
            "ade_improvement": 0.0,
            "fde_improvement": 0.0,
            "relative_ade_gain": 0.0,
            "refinement_case_score": 0.0,
        }

    initial_xy = initial[:num_steps, :2]
    refined_xy = refined[:num_steps, :2]
    expert_xy = expert[:num_steps, :2]
    initial_errors = np.linalg.norm(initial_xy - expert_xy, axis=-1)
    refined_errors = np.linalg.norm(refined_xy - expert_xy, axis=-1)
    initial_ade = float(initial_errors.mean())
    refined_ade = float(refined_errors.mean())
    initial_fde = float(initial_errors[-1])
    refined_fde = float(refined_errors[-1])
    ade_improvement = initial_ade - refined_ade
    fde_improvement = initial_fde - refined_fde
    relative_ade_gain = ade_improvement / max(initial_ade, 1e-6)

    # Prefer cases where the initial query is visibly wrong, the refined query
    # moves closer to expert, and the final trajectory is not still far away.
    refinement_case_score = (
        max(ade_improvement, 0.0)
        * (1.0 + max(relative_ade_gain, 0.0))
        * (1.0 + min(initial_ade, 8.0) / 8.0)
        / (1.0 + refined_ade)
    )
    failure_case_score = (
        refined_ade
        + 0.5 * refined_fde
        + 2.0 * max(-ade_improvement, 0.0)
        + (0.5 if ade_improvement < 0.0 else 0.0)
    )

    return {
        "initial_ade": initial_ade,
        "refined_ade": refined_ade,
        "initial_fde": initial_fde,
        "refined_fde": refined_fde,
        "ade_improvement": ade_improvement,
        "fde_improvement": fde_improvement,
        "relative_ade_gain": relative_ade_gain,
        "refinement_case_score": float(refinement_case_score),
        "failure_case_score": float(failure_case_score),
    }


def _trajectory_path_length(trajectory: np.ndarray) -> float:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if len(trajectory) < 2:
        return float(np.linalg.norm(trajectory[-1, :2])) if len(trajectory) == 1 else 0.0
    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), trajectory[:, :2]], axis=0)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())


def _count_nearby_agents(scene: Scene) -> Tuple[int, int]:
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    boxes = np.asarray(frame.annotations.boxes)
    if boxes.size == 0:
        return 0, 0
    x = boxes[:, BoundingBoxIndex.X]
    y = boxes[:, BoundingBoxIndex.Y]
    forward = (x > 0.0) & (x < 30.0) & (np.abs(y) < 7.5)
    nearby = (x > -10.0) & (x < 30.0) & (np.abs(y) < 12.0)
    return int(forward.sum()), int(nearby.sum())


def _classify_scene_type(scene: Scene, gt_trajectory: np.ndarray) -> Dict[str, Any]:
    gt_trajectory = np.asarray(gt_trajectory, dtype=np.float32)
    if gt_trajectory.size == 0:
        return {
            "scenario_type": "unknown",
            "final_lateral": 0.0,
            "final_heading": 0.0,
            "path_length": 0.0,
            "forward_agents": 0,
            "nearby_agents": 0,
        }

    final_x, final_y, final_heading = gt_trajectory[-1, :3]
    path_length = _trajectory_path_length(gt_trajectory)
    forward_agents, nearby_agents = _count_nearby_agents(scene)
    ego_speed = float(
        np.linalg.norm(scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status.ego_velocity[:2])
    )

    if (path_length < 7.0 or ego_speed < 2.0) and forward_agents >= 2:
        scenario_type = "traffic_jam"
    elif final_heading > 0.28 or final_y > 2.8:
        scenario_type = "left_turn"
    elif final_heading < -0.28 or final_y < -2.8:
        scenario_type = "right_turn"
    elif abs(final_heading) < 0.22 and abs(final_y) < 2.5 and final_x > 8.0:
        scenario_type = "straight"
    else:
        scenario_type = "straight" if abs(final_y) <= 3.5 else "unknown"

    return {
        "scenario_type": scenario_type,
        "final_lateral": float(final_y),
        "final_heading": float(final_heading),
        "path_length": float(path_length),
        "ego_speed": ego_speed,
        "forward_agents": forward_agents,
        "nearby_agents": nearby_agents,
    }


def _collect_sample_visualization(
    token: str,
    scene: Scene,
    trajectory,
    predictions: Dict[str, torch.Tensor],
    vis_dict: Dict[str, Any],
    agent: IDOLAgent,
    baseline_trajectory=None,
) -> Dict[str, Any]:
    selected_idx = int(_tensor_to_numpy(vis_dict["selected_candidate_idx"])[0])
    bev_hw = tuple(int(v) for v in _tensor_to_numpy(vis_dict["bev_hw"]).tolist())
    num_poses = agent.config.trajectory_sampling.num_poses
    gt_trajectory = scene.get_future_trajectory(num_trajectory_frames=num_poses).poses
    scenario_metrics = _classify_scene_type(scene, gt_trajectory)

    future_bev_states = np.zeros((0, bev_hw[0] * bev_hw[1], 0), dtype=np.float32)
    future_transition_maps = np.zeros((0, bev_hw[0], bev_hw[1]), dtype=np.float32)
    if vis_dict.get("future_bev_states"):
        final_iter_states = vis_dict["future_bev_states"][-1]
        selected_states = [
            _select_candidate(state, selected_idx)
            for state in final_iter_states
        ]
        future_bev_states = np.stack(selected_states, axis=0)
        future_transition_maps = compute_transition_heatmap(
            future_bev_states,
            mode="l2",
            bev_hw=bev_hw,
            normalize=True,
        )
    else:
        warnings.warn(f"No future BEV states were collected for token={token}")

    idm_spatial_maps = np.zeros((0, bev_hw[0] * bev_hw[1], 0), dtype=np.float32)
    idm_spatial_heatmaps = np.zeros((0, bev_hw[0], bev_hw[1]), dtype=np.float32)
    idm_spatial_map_avg = np.zeros((bev_hw[0], bev_hw[1]), dtype=np.float32)
    if vis_dict.get("idm_spatial_maps"):
        final_iter_idm = _tensor_to_numpy(vis_dict["idm_spatial_maps"][-1])
        idm_spatial_maps = final_iter_idm[:, 0, selected_idx]
        idm_spatial_heatmaps = np.stack(
            [
                normalize_heatmap(tensor_to_bev_map(spatial_map, mode="l2", bev_hw=bev_hw))
                for spatial_map in idm_spatial_maps
            ],
            axis=0,
        )
        idm_spatial_map_avg = normalize_heatmap(np.mean(idm_spatial_heatmaps, axis=0))
    else:
        warnings.warn(f"No IDM spatial maps were collected for token={token}")

    idm_attention_maps = np.zeros((0, bev_hw[0], bev_hw[1]), dtype=np.float32)
    idm_attention_map_final = np.zeros((bev_hw[0], bev_hw[1]), dtype=np.float32)
    if vis_dict.get("idm_spatial_attention"):
        attention_heatmaps = []
        for iter_attention in vis_dict["idm_spatial_attention"]:
            iter_attention_np = _tensor_to_numpy(iter_attention)
            selected_attention = iter_attention_np[0, selected_idx, 0]
            attention_heatmaps.append(
                normalize_heatmap(tensor_to_bev_map(selected_attention, mode="mean", bev_hw=bev_hw))
            )
        idm_attention_maps = np.stack(attention_heatmaps, axis=0).astype(np.float32)
        idm_attention_map_final = idm_attention_maps[-1]

    evidence_step_idx, idm_transition_alignment, idm_spatial_alignment = _select_evidence_step(
        future_transition_maps,
        idm_spatial_heatmaps,
    )
    if future_transition_maps.size > 0 and idm_attention_maps.size > 0:
        idm_attention_alignment = compute_heatmap_alignment(
            future_transition_maps[evidence_step_idx],
            idm_attention_map_final,
        )
    else:
        idm_attention_alignment = {}

    idm_global_features = np.zeros((0, 0), dtype=np.float32)
    if vis_dict.get("idm_global_features"):
        final_iter_global = _tensor_to_numpy(vis_dict["idm_global_features"][-1])
        idm_global_features = final_iter_global[:, 0, selected_idx]

    refinement_trajectories: List[np.ndarray] = []
    for traj_candidates in vis_dict.get("refinement_trajectories", []):
        refinement_trajectories.append(_select_candidate(traj_candidates, selected_idx))
    if len(refinement_trajectories) < 2:
        warnings.warn(
            f"Token={token} has only {len(refinement_trajectories)} refinement trajectory snapshots. "
            "Figure 4(d) will show the available initial/refined stages."
        )

    final_trajectory = _tensor_to_numpy(predictions["trajectory"])[0]
    all_trajectory = _tensor_to_numpy(predictions["all_trajectory"])[0]
    final_rewards = _tensor_to_numpy(predictions["final_rewards"])[0]
    im_rewards = _tensor_to_numpy(predictions["im_rewards"])[0]
    if len(refinement_trajectories) >= 2:
        refinement_metrics = _trajectory_error_metrics(
            refinement_trajectories[0],
            refinement_trajectories[-1],
            gt_trajectory,
        )
    else:
        refinement_metrics = _trajectory_error_metrics(
            final_trajectory,
            final_trajectory,
            gt_trajectory,
        )

    metadata = {
        "token": token,
        "log_name": scene.scene_metadata.log_name,
        "scene_token": scene.scene_metadata.scene_token,
        "map_name": scene.scene_metadata.map_name,
        "selected_candidate_idx": selected_idx,
        "bev_hw": list(bev_hw),
        "num_closed_loop_iters": int(getattr(agent.config, "num_closed_loop_iters", 1)),
        "num_fut_timestep": int(getattr(agent.config, "num_fut_timestep", 1)),
        "scenario_type": scenario_metrics["scenario_type"],
        "notes": (
            "future_bev_states/idm_spatial_maps are sliced at the final selected "
            "candidate index from forward_test_with_vis. idm_evidence_step_idx selects "
            "the adjacent BEV pair with strongest IDM/transition alignment."
        ),
    }

    return {
        "token": token,
        "scene": scene,
        "metadata": metadata,
        "bev_hw": bev_hw,
        "current_bev": _tensor_to_numpy(vis_dict["current_bev"])[0],
        "future_bev_states": future_bev_states,
        "future_transition_maps": future_transition_maps,
        "idm_spatial_maps": idm_spatial_maps,
        "idm_spatial_heatmaps": idm_spatial_heatmaps,
        "idm_spatial_map_avg": idm_spatial_map_avg,
        "idm_attention_maps": idm_attention_maps,
        "idm_attention_map_final": idm_attention_map_final,
        "idm_transition_alignment": idm_transition_alignment,
        "idm_spatial_alignment": idm_spatial_alignment,
        "idm_attention_alignment": idm_attention_alignment,
        "idm_evidence_step_idx": evidence_step_idx,
        "idm_evidence_score": heatmap_alignment_score(idm_spatial_alignment) if idm_spatial_alignment else 0.0,
        "idm_global_features": idm_global_features,
        "refinement_trajectories": refinement_trajectories,
        "refinement_metrics": refinement_metrics,
        "scenario_metrics": scenario_metrics,
        "final_trajectory": final_trajectory,
        "idol_trajectory": trajectory.poses,
        "gt_trajectory": gt_trajectory,
        "baseline_trajectory": None if baseline_trajectory is None else baseline_trajectory.poses,
        "selected_candidate_idx": selected_idx,
        "all_trajectory": all_trajectory,
        "final_rewards": final_rewards,
        "im_rewards": im_rewards,
    }


def _save_npz(sample: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refinement = np.asarray(sample["refinement_trajectories"], dtype=np.float32)
    fields = {
        "current_bev": sample["current_bev"],
        "future_bev_states": sample["future_bev_states"],
        "future_transition_maps": sample["future_transition_maps"],
        "idm_spatial_maps": sample["idm_spatial_maps"],
        "idm_spatial_heatmaps": sample["idm_spatial_heatmaps"],
        "idm_spatial_map_avg": sample["idm_spatial_map_avg"],
        "idm_attention_maps": sample["idm_attention_maps"],
        "idm_attention_map_final": sample["idm_attention_map_final"],
        "idm_global_features": sample["idm_global_features"],
        "refinement_trajectories": refinement,
        "refinement_metrics": np.asarray(json.dumps(sample["refinement_metrics"], indent=2)),
        "scenario_metrics": np.asarray(json.dumps(sample["scenario_metrics"], indent=2)),
        "final_trajectory": sample["final_trajectory"],
        "gt_trajectory": sample["gt_trajectory"],
        "selected_candidate_idx": np.asarray(sample["selected_candidate_idx"], dtype=np.int64),
        "idm_evidence_step_idx": np.asarray(sample["idm_evidence_step_idx"], dtype=np.int64),
        "idm_evidence_score": np.asarray(sample["idm_evidence_score"], dtype=np.float32),
        "idm_alignment_metrics": np.asarray(
            json.dumps(
                {
                    "transition_alignment": sample["idm_transition_alignment"],
                    "spatial_alignment": sample["idm_spatial_alignment"],
                    "attention_alignment": sample["idm_attention_alignment"],
                },
                indent=2,
            )
        ),
        "final_rewards": sample["final_rewards"],
        "im_rewards": sample["im_rewards"],
        "all_trajectory": sample["all_trajectory"],
        "metadata": np.asarray(json.dumps(sample["metadata"], indent=2)),
    }
    if sample.get("baseline_trajectory") is not None:
        fields["baseline_trajectory"] = sample["baseline_trajectory"]
    np.savez_compressed(output_path, **fields)


def _write_report(samples: Sequence[Dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "IDOL visualization report",
        "",
        "Generated files:",
        "- figure3_qualitative_trajectories.png/pdf: qualitative BEV trajectory comparison.",
        "- figure4_idm_refinement.png/pdf: best-aligned sample IDM evidence/refinement four-panel figure.",
        "- samples/sample_*_vis_data.npz: detached per-sample intermediate tensors.",
        "- samples/sample_*_fig3.png/pdf and sample_*_fig4.png/pdf: per-sample panels.",
        "",
        "Intermediate tensor provenance:",
        "- current_bev: IDOLModel.extract_trajectory_feature -> flatten_bev_feature after backbone and key/value positional embedding.",
        "- future_bev_states: IDOLModel.extract_reward_feature closed-loop latent world model bev_feat_list, sliced at final selected candidate.",
        "- future_transition_maps: mean/L2 difference between adjacent future_bev_states, computed by visualization_utils.compute_transition_heatmap.",
        "- idm_spatial_maps: IDOLModel._inverse_dynamics_prediction output spatial_dynamics from InverseDynamicsModel.",
        "- idm_spatial_map_avg: L2 channel reduction of idm_spatial_maps followed by percentile normalization and averaging.",
        "- idm_attention_map_final: final closed-loop ego-query cross-attention over IDM spatial dynamics.",
        "- idm_alignment_metrics: Pearson/cosine/top-region agreement between latent BEV transition and IDM response/attention.",
        "- idm_global_features: IDOLModel._inverse_dynamics_prediction output global_dynamics.",
        "- refinement_trajectories: IDOLModel.forward_test_with_vis decodes initial/refined queries with the existing offset decoder, after final prediction is computed.",
        "- refinement_metrics: ADE/FDE from initial and final refined query trajectories to expert; used to rank paper-worthy refinement cases.",
        "- final_trajectory: selected trajectory from IDOLModel.forward_test_with_vis.",
        "- gt_trajectory: Scene.get_future_trajectory using the agent trajectory_sampling.num_poses.",
        "",
        f"Command figure mode: {args.figure}",
        f"Checkpoint: {args.checkpoint_path}",
        f"Baseline checkpoint: {args.baseline_checkpoint_path or 'none'}",
        f"Split: {args.split}",
        "",
        "Samples:",
    ]
    for idx, sample in enumerate(samples):
        metadata = sample["metadata"]
        spatial = sample.get("idm_spatial_alignment", {})
        attention = sample.get("idm_attention_alignment", {})
        refinement = sample.get("refinement_metrics", {})
        scenario = sample.get("scenario_metrics", {})
        lines.append(
            f"- {idx:03d} token={metadata['token']} log={metadata['log_name']} "
            f"scenario={scenario.get('scenario_type', metadata.get('scenario_type', 'unknown'))} "
            f"selected_candidate_idx={metadata['selected_candidate_idx']} bev_hw={metadata['bev_hw']} "
            f"idm_step={sample.get('idm_evidence_step_idx', 0)} "
            f"spatial_corr={spatial.get('pearson', 0.0):.3f} "
            f"spatial_top_recall={spatial.get('topk_recall', 0.0):.3f} "
            f"attention_energy_on_change={attention.get('response_energy_in_reference_top', 0.0):.3f} "
            f"path_length={scenario.get('path_length', 0.0):.3f} "
            f"final_lateral={scenario.get('final_lateral', 0.0):.3f} "
            f"final_heading={scenario.get('final_heading', 0.0):.3f} "
            f"forward_agents={scenario.get('forward_agents', 0)} "
            f"initial_ade={refinement.get('initial_ade', 0.0):.3f} "
            f"refined_ade={refinement.get('refined_ade', 0.0):.3f} "
            f"ade_improvement={refinement.get('ade_improvement', 0.0):.3f} "
            f"refinement_case_score={refinement.get('refinement_case_score', 0.0):.3f} "
            f"failure_case_score={refinement.get('failure_case_score', 0.0):.3f}"
        )
    (output_dir / "visualization_report.txt").write_text("\n".join(lines))


def _select_idm_evidence_sample(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("No samples available.")
    return max(samples, key=lambda sample: float(sample.get("idm_evidence_score", 0.0)))


def _select_refinement_cases(
    samples: Sequence[Dict[str, Any]],
    num_cases: int,
    min_initial_ade: float,
    max_refined_ade: float,
    min_ade_improvement: float,
) -> List[Dict[str, Any]]:
    filtered = []
    for sample in samples:
        metrics = sample.get("refinement_metrics", {})
        if metrics.get("initial_ade", 0.0) < min_initial_ade:
            continue
        if metrics.get("refined_ade", 0.0) > max_refined_ade:
            continue
        if metrics.get("ade_improvement", 0.0) < min_ade_improvement:
            continue
        filtered.append(sample)

    if len(filtered) < num_cases:
        LOGGER.warning(
            "Only %d samples met the refinement thresholds; filling from top-scored candidates.",
            len(filtered),
        )
        selected_tokens = {sample["token"] for sample in filtered}
        fallback = [sample for sample in samples if sample["token"] not in selected_tokens]
        filtered.extend(fallback)

    ranked = sorted(
        filtered,
        key=lambda sample: float(sample.get("refinement_metrics", {}).get("refinement_case_score", 0.0)),
        reverse=True,
    )
    return ranked[:num_cases]


def _normalize_scenario_name(name: str) -> str:
    aliases = {
        "jam": "traffic_jam",
        "traffic": "traffic_jam",
        "trafficjam": "traffic_jam",
        "traffic_jam": "traffic_jam",
        "left": "left_turn",
        "left_turn": "left_turn",
        "right": "right_turn",
        "right_turn": "right_turn",
        "straight": "straight",
        "go_straight": "straight",
    }
    return aliases.get(name.strip().lower(), name.strip().lower())


def _select_scenario_cases(
    samples: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    cases_per_scenario: int,
    min_initial_ade: float,
    max_refined_ade: float,
    min_ade_improvement: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_tokens = set()
    normalized_categories = [_normalize_scenario_name(category) for category in categories]

    for category in normalized_categories:
        candidates = []
        for sample in samples:
            scenario_type = sample.get("scenario_metrics", {}).get("scenario_type", "unknown")
            if scenario_type != category:
                continue
            metrics = sample.get("refinement_metrics", {})
            if metrics.get("initial_ade", 0.0) < min_initial_ade:
                continue
            if metrics.get("refined_ade", 0.0) > max_refined_ade:
                continue
            if metrics.get("ade_improvement", 0.0) < min_ade_improvement:
                continue
            candidates.append(sample)

        if len(candidates) < cases_per_scenario:
            LOGGER.warning(
                "Only %d %s samples met thresholds; filling this bucket from all %s samples.",
                len(candidates),
                category,
                category,
            )
            candidate_tokens = {sample["token"] for sample in candidates}
            candidates.extend(
                sample
                for sample in samples
                if sample["token"] not in candidate_tokens
                and sample.get("scenario_metrics", {}).get("scenario_type", "unknown") == category
            )

        ranked = sorted(
            candidates,
            key=lambda sample: float(sample.get("refinement_metrics", {}).get("refinement_case_score", 0.0)),
            reverse=True,
        )
        for sample in ranked:
            if sample["token"] in selected_tokens:
                continue
            selected.append(sample)
            selected_tokens.add(sample["token"])
            if sum(
                item.get("scenario_metrics", {}).get("scenario_type") == category
                for item in selected
            ) >= cases_per_scenario:
                break

    if not selected:
        LOGGER.warning("No scenario cases selected; falling back to top refinement cases.")
        selected = _select_refinement_cases(
            samples,
            num_cases=len(normalized_categories) * cases_per_scenario,
            min_initial_ade=min_initial_ade,
            max_refined_ade=max_refined_ade,
            min_ade_improvement=min_ade_improvement,
        )

    return selected


def _select_failure_cases(
    samples: Sequence[Dict[str, Any]],
    num_cases: int,
    min_refined_ade: float,
    min_worsening: float,
) -> List[Dict[str, Any]]:
    filtered = []
    for sample in samples:
        metrics = sample.get("refinement_metrics", {})
        refined_ade = metrics.get("refined_ade", 0.0)
        ade_improvement = metrics.get("ade_improvement", 0.0)
        if refined_ade >= min_refined_ade or ade_improvement <= -min_worsening:
            filtered.append(sample)

    if len(filtered) < num_cases:
        LOGGER.warning(
            "Only %d samples met failure thresholds; filling from top failure-score candidates.",
            len(filtered),
        )
        selected_tokens = {sample["token"] for sample in filtered}
        filtered.extend(sample for sample in samples if sample["token"] not in selected_tokens)

    ranked = sorted(
        filtered,
        key=lambda sample: float(sample.get("refinement_metrics", {}).get("failure_case_score", 0.0)),
        reverse=True,
    )
    return ranked[:num_cases]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IDOL paper visualizations for IDOL/NAVSIM.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the IDOL/IDOL checkpoint.")
    parser.add_argument("--output_dir", default="outputs/idol_visualization", help="Directory for figures and npz files.")
    parser.add_argument("--split", default="navtest", help="Logical split: navtest, test, val, mini, trainval, etc.")
    parser.add_argument("--data_split", default=None, help="Override data folder under OPENSCENE_DATA_ROOT/navsim_logs.")
    parser.add_argument("--scene_filter", default=None, help="Override scene_filter yaml name, e.g. navtest.")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of samples to visualize.")
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=None,
        help="Number of candidate scenes to score when selecting refinement cases.",
    )
    parser.add_argument(
        "--select_refinement_cases",
        action="store_true",
        help="Rank candidates by initial-to-refined trajectory improvement and only save the top cases.",
    )
    parser.add_argument(
        "--select_scenario_cases",
        action="store_true",
        help="Select cases from maneuver buckets such as traffic_jam, left_turn, right_turn, and straight.",
    )
    parser.add_argument(
        "--select_failure_cases",
        action="store_true",
        help="Rank candidates by poor refined trajectory quality or refinement degradation.",
    )
    parser.add_argument("--num_failure_cases", type=int, default=2, help="Number of failure cases to save.")
    parser.add_argument(
        "--min_failure_refined_ade",
        type=float,
        default=2.0,
        help="Minimum refined ADE for a sample to qualify as a failure.",
    )
    parser.add_argument(
        "--min_failure_worsening",
        type=float,
        default=0.05,
        help="Minimum ADE degradation for a sample to qualify as a failure.",
    )
    parser.add_argument(
        "--scenario_categories",
        nargs="*",
        default=["traffic_jam", "left_turn", "right_turn", "straight"],
        help="Scenario buckets to select when --select_scenario_cases is used.",
    )
    parser.add_argument(
        "--cases_per_scenario",
        type=int,
        default=2,
        help="Number of cases to save per scenario bucket.",
    )
    parser.add_argument(
        "--num_selected_cases",
        type=int,
        default=None,
        help="Number of top refinement cases to save; defaults to --num_samples.",
    )
    parser.add_argument(
        "--min_initial_ade",
        type=float,
        default=0.0,
        help="Minimum initial-query ADE for refinement case selection.",
    )
    parser.add_argument(
        "--max_refined_ade",
        type=float,
        default=1e9,
        help="Maximum refined-query ADE for refinement case selection.",
    )
    parser.add_argument(
        "--min_ade_improvement",
        type=float,
        default=0.0,
        help="Minimum ADE improvement for refinement case selection.",
    )
    parser.add_argument("--sample_tokens", nargs="*", default=None, help="Optional token list, space or comma separated.")
    parser.add_argument("--log_names", nargs="*", default=None, help="Optional log name list, space or comma separated.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--figure",
        choices=["traj", "idm", "failure", "all"],
        default="all",
        help="Which figure(s) to generate.",
    )
    parser.set_defaults(save_pdf=True)
    parser.add_argument("--save_pdf", dest="save_pdf", action="store_true", help="Save PDF copies for paper figures.")
    parser.add_argument("--no_save_pdf", dest="save_pdf", action="store_false", help="Disable PDF copies.")
    parser.add_argument("--save_vis", action="store_true", help="Save per-sample npz intermediate tensors.")
    parser.add_argument("--baseline_checkpoint_path", default=None, help="Optional baseline checkpoint for Figure 3.")
    parser.add_argument("--navsim_log_path", default=None, help="Override navsim log path.")
    parser.add_argument("--sensor_blobs_path", default=None, help="Override sensor blobs path.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for sampling tokens.")
    parser.add_argument(
        "--font_family",
        default=None,
        help="Optional matplotlib font family for all generated text, e.g. 'Times New Roman'.",
    )
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        help="Optional IDOLConfig override key=value; repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    _configure_matplotlib_font(args.font_family)
    args.checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path, "--checkpoint_path")
    if args.baseline_checkpoint_path:
        args.baseline_checkpoint_path = _resolve_checkpoint_path(
            args.baseline_checkpoint_path,
            "--baseline_checkpoint_path",
        )

    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_tokens = _parse_tokens(args.sample_tokens)
    log_names = _parse_tokens(args.log_names)
    data_split, scene_filter_name = _resolve_split(args.split, args.data_split, args.scene_filter)

    openscene_root = Path(os.environ.get("OPENSCENE_DATA_ROOT", "dataset"))
    navsim_log_path = Path(args.navsim_log_path or (openscene_root / "navsim_logs" / data_split))
    sensor_blobs_path = Path(args.sensor_blobs_path or (openscene_root / "sensor_blobs" / data_split))
    if not navsim_log_path.exists():
        raise FileNotFoundError(f"NAVSIM log path not found: {navsim_log_path}")
    if not sensor_blobs_path.exists():
        raise FileNotFoundError(f"Sensor blobs path not found: {sensor_blobs_path}")

    candidate_count = args.candidate_pool_size if args.candidate_pool_size is not None else args.num_samples
    exhaustive_selector = args.select_scenario_cases or args.select_failure_cases
    filter_sample_limit = 0 if exhaustive_selector and not sample_tokens else candidate_count
    scene_filter = _load_scene_filter(scene_filter_name, sample_tokens, filter_sample_limit, log_names)
    LOGGER.info("Loading scenes from %s with scene_filter=%s", navsim_log_path, scene_filter_name)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )

    tokens = list(scene_loader.tokens)
    if sample_tokens:
        missing = [token for token in sample_tokens if token not in tokens]
        if missing:
            LOGGER.warning("Skipping %d requested tokens not present in SceneLoader: %s", len(missing), missing[:8])
        tokens_to_render = [token for token in sample_tokens if token in tokens]
    else:
        if args.seed is not None:
            rng = random.Random(args.seed)
            tokens_to_render = rng.sample(tokens, min(candidate_count, len(tokens)))
        else:
            tokens_to_render = tokens[:candidate_count]
    if not tokens_to_render:
        raise RuntimeError("No tokens selected for visualization.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if str(device) != args.device:
        LOGGER.warning("Requested device=%s but CUDA is unavailable; using cpu.", args.device)

    LOGGER.info("Loading IDOL checkpoint: %s", args.checkpoint_path)
    agent = _build_agent(args.checkpoint_path, device, args.config_override)

    baseline_agent = None
    if args.baseline_checkpoint_path:
        LOGGER.info("Loading baseline checkpoint: %s", args.baseline_checkpoint_path)
        baseline_agent = _build_agent(args.baseline_checkpoint_path, device, args.config_override)

    samples: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens_to_render):
        LOGGER.info("[%d/%d] Rendering token=%s", idx + 1, len(tokens_to_render), token)
        scene = scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        trajectory, predictions, vis_dict = agent.compute_trajectory_with_idol_vis(agent_input)
        baseline_trajectory = None
        if baseline_agent is not None:
            baseline_trajectory = baseline_agent.compute_trajectory(agent_input)

        sample = _collect_sample_visualization(
            token=token,
            scene=scene,
            trajectory=trajectory,
            predictions=predictions,
            vis_dict=vis_dict,
            agent=agent,
            baseline_trajectory=baseline_trajectory,
        )
        samples.append(sample)

    output_samples = samples
    if args.select_failure_cases:
        output_samples = _select_failure_cases(
            samples,
            num_cases=args.num_failure_cases,
            min_refined_ade=args.min_failure_refined_ade,
            min_worsening=args.min_failure_worsening,
        )
        LOGGER.info(
            "Selected %d failure cases from %d candidates.",
            len(output_samples),
            len(samples),
        )
    elif args.select_scenario_cases:
        output_samples = _select_scenario_cases(
            samples,
            categories=args.scenario_categories,
            cases_per_scenario=args.cases_per_scenario,
            min_initial_ade=args.min_initial_ade,
            max_refined_ade=args.max_refined_ade,
            min_ade_improvement=args.min_ade_improvement,
        )
        LOGGER.info(
            "Selected %d scenario-diverse cases from %d candidates.",
            len(output_samples),
            len(samples),
        )
    elif args.select_refinement_cases:
        num_selected_cases = args.num_selected_cases or args.num_samples
        output_samples = _select_refinement_cases(
            samples,
            num_cases=num_selected_cases,
            min_initial_ade=args.min_initial_ade,
            max_refined_ade=args.max_refined_ade,
            min_ade_improvement=args.min_ade_improvement,
        )
        LOGGER.info(
            "Selected %d refinement cases from %d candidates.",
            len(output_samples),
            len(samples),
        )

    for idx, sample in enumerate(output_samples):
        sample_id = f"sample_{idx:03d}_{_sanitize(sample['token'])}"
        if args.save_vis:
            _save_npz(sample, sample_dir / f"{sample_id}_vis_data.npz")

        if args.figure in ("traj", "all"):
            fig3_sample = make_single_sample_trajectory_figure(sample)
            save_figure(fig3_sample, sample_dir / f"{sample_id}_fig3.png", save_pdf=args.save_pdf)

        if args.figure in ("idm", "all"):
            fig4_sample = make_figure4_idm_refinement(sample)
            save_figure(fig4_sample, sample_dir / f"{sample_id}_fig4.png", save_pdf=args.save_pdf)

    if args.figure in ("traj", "all"):
        fig3 = make_figure3_qualitative(output_samples)
        save_figure(fig3, output_dir / "figure3_qualitative_trajectories.png", save_pdf=args.save_pdf)

    if args.figure in ("idm", "all"):
        if args.select_scenario_cases or args.select_refinement_cases:
            fig4_sample = output_samples[0]
            LOGGER.info(
                "Using token=%s for aggregate Figure 4 (scenario=%s, refinement case score=%.3f).",
                fig4_sample["metadata"]["token"],
                fig4_sample.get("scenario_metrics", {}).get("scenario_type", "unknown"),
                fig4_sample.get("refinement_metrics", {}).get("refinement_case_score", 0.0),
            )
        else:
            fig4_sample = _select_idm_evidence_sample(output_samples)
            LOGGER.info(
                "Using token=%s for aggregate Figure 4 (IDM evidence score=%.3f).",
                fig4_sample["metadata"]["token"],
                fig4_sample.get("idm_evidence_score", 0.0),
            )
        fig4 = make_figure4_idm_refinement(fig4_sample)
        save_figure(fig4, output_dir / "figure4_idm_refinement.png", save_pdf=args.save_pdf)

    if args.figure in ("failure", "all"):
        failure_fig = make_failure_cases_figure(output_samples, max_cases=args.num_failure_cases)
        save_figure(failure_fig, output_dir / "figure_failure_cases.png", save_pdf=args.save_pdf)

    _write_report(output_samples, output_dir, args)
    LOGGER.info("Done. Outputs saved under: %s", output_dir)


if __name__ == "__main__":
    main()
