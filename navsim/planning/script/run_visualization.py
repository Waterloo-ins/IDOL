"""


    ${output_dir}/figures/<token>/<plot_name>.png



    python ./navsim/planning/script/run_visualization.py \
        agent=IDOL_agent \
        agent.checkpoint_path="/path/to.ckpt" \
        experiment_name=visualization/IDOL/default \
        split=test \
        scene_filter=navtest \
        num_scenes=5 \
        plot_types=[bev_frame,bev_agent,bev_candidates,cameras,cameras_annotations]

"""
from __future__ import annotations

import logging
import random
import traceback
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import (
    plot_bev_frame,
    plot_bev_lidar_frame,
    plot_bev_with_agent,
    plot_bev_with_agent_and_traj_candidates,
    plot_cameras_frame,
    plot_cameras_frame_full,
    plot_cameras_frame_with_annotations,
    plot_cameras_frame_with_lidar,
    plot_front_view_frame,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/visualization"
CONFIG_NAME = "default_run_visualization"


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int = 150) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_for_token(
    token: str,
    scene: Scene,
    agent: AbstractAgent,
    plot_types: List[str],
    figures_dir: Path,
) -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    token_dir = figures_dir / token
    token_dir.mkdir(parents=True, exist_ok=True)

    for plot_type in plot_types:
        try:
            if plot_type == "bev_frame":
                fig, _ = plot_bev_frame(scene, frame_idx)
                _save_fig(fig, token_dir / "bev_frame.png")

            elif plot_type == "bev_agent":
                fig, _ = plot_bev_with_agent(scene, agent)
                _save_fig(fig, token_dir / "bev_agent.png")

            elif plot_type == "bev_candidates":
                if not hasattr(agent, "compute_trajectory_with_vis"):
                    logger.warning(
                        f"Agent {agent.name()} has no `compute_trajectory_with_vis`, "
                        f"skipping `bev_candidates` for token {token}."
                    )
                    continue
                fig, _ = plot_bev_with_agent_and_traj_candidates(scene, agent)
                _save_fig(fig, token_dir / "bev_candidates.png")

            elif plot_type == "cameras":
                fig, _ = plot_cameras_frame(scene, frame_idx)
                _save_fig(fig, token_dir / "cameras.png")

            elif plot_type == "cameras_annotations":
                fig, _ = plot_cameras_frame_with_annotations(scene, frame_idx)
                _save_fig(fig, token_dir / "cameras_annotations.png")

            elif plot_type == "cameras_lidar":
                fig, _ = plot_cameras_frame_with_lidar(scene, frame_idx)
                _save_fig(fig, token_dir / "cameras_lidar.png")

            elif plot_type == "cameras_full":
                fig, _ = plot_cameras_frame_full(scene, frame_idx)
                _save_fig(fig, token_dir / "cameras_full.png")

            elif plot_type == "bev_lidar":
                fig, _ = plot_bev_lidar_frame(scene, frame_idx)
                _save_fig(fig, token_dir / "bev_lidar.png")

            elif plot_type == "front_view":
                fig, _ = plot_front_view_frame(scene, frame_idx)
                _save_fig(fig, token_dir / "front_view.png")

            else:
                logger.warning(f"Unknown plot type: {plot_type}, skipping.")
        except Exception:
            logger.warning(f"Failed to render plot_type={plot_type} for token={token}:")
            traceback.print_exc()


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    output_dir = Path(cfg.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Visualization output dir: {figures_dir}")

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.is_eval = True

    scene_filter: SceneFilter = instantiate(cfg.scene_filter)

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )

    all_tokens: List[str] = list(scene_loader.tokens)
    if not all_tokens:
        logger.error("No tokens matched by the scene_filter. Exiting.")
        return

    token_list_cfg = cfg.get("token_list", None)
    if token_list_cfg:
        tokens_to_plot = [t for t in token_list_cfg if t in all_tokens]
        missing = [t for t in token_list_cfg if t not in all_tokens]
        if missing:
            logger.warning(f"{len(missing)} tokens not found in loader and will be skipped.")
    else:
        num_scenes = int(cfg.get("num_scenes", 5))
        num_scenes = max(1, min(num_scenes, len(all_tokens)))
        seed = cfg.get("seed", None)
        if seed is not None:
            random.seed(int(seed))
            tokens_to_plot = random.sample(all_tokens, num_scenes)
        else:
            tokens_to_plot = all_tokens[:num_scenes]

    plot_types_cfg = cfg.get("plot_types", None)
    if isinstance(plot_types_cfg, (list, ListConfig)):
        plot_types = list(plot_types_cfg)
    else:
        plot_types = [
            "bev_frame",
            "bev_agent",
            "bev_candidates",
            "cameras",
            "cameras_annotations",
            "cameras_full",
            "bev_lidar",
        ]
    logger.info(f"Rendering plot_types={plot_types} for {len(tokens_to_plot)} tokens")

    for idx, token in enumerate(tokens_to_plot):
        logger.info(f"[{idx + 1}/{len(tokens_to_plot)}] Rendering token={token}")
        try:
            scene = scene_loader.get_scene_from_token(token)
            _plot_for_token(token, scene, agent, plot_types, figures_dir)
        except Exception:
            logger.warning(f"Failed to load/plot scene for token={token}:")
            traceback.print_exc()

    with open(figures_dir / "tokens.txt", "w") as f:
        f.write("\n".join(tokens_to_plot))
    with open(figures_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    logger.info(f"Done. Figures saved under: {figures_dir}")


if __name__ == "__main__":
    main()
