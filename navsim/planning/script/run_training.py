import os
import sys
import types
from typing import Tuple

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    pass
except ImportError as exc:
    if "huggingface-hub" not in str(exc):
        raise
    sys.modules.pop("transformers", None)
    transformers_stub = types.ModuleType("transformers")

    class _UnavailableTransformersObject:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers is disabled for NAVSIM training startup.")

    transformers_stub.AutoModel = _UnavailableTransformersObject
    transformers_stub.AutoTokenizer = _UnavailableTransformersObject
    transformers_stub.__version__ = "0.0.0"
    sys.modules["transformers"] = transformers_stub

import hydra
from hydra.utils import instantiate
import logging, torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.loader_utils import build_training_loaders
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent

from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        # train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
        train_scene_filter.log_names = list(set(train_scene_filter.log_names) & set(cfg.train_logs))
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        # val_scene_filter.log_names = [l for l in val_scene_filter.log_names if l in cfg.val_logs]
        val_scene_filter.log_names = list(set(val_scene_filter.log_names) & set(cfg.val_logs))
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    train_debug = cfg.train_debug if hasattr(cfg, "train_debug") else False

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        train_debug=train_debug,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    use_fut_frames = agent.config.use_fut_frames if hasattr(agent.config, "use_fut_frames") else False
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get('seed', 0))
    logger.info(f"Global Seed set to {seed}")
    pl.seed_everything(seed, workers=True)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert cfg.force_cache_computation==False, "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader, val_dataloader = build_training_loaders(train_data, val_data, cfg.dataloader)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("DataLoader workers: train=%d, validation=%d; pin_memory: train=%s, validation=%s",
                train_dataloader.num_workers, val_dataloader.num_workers,
                train_dataloader.pin_memory, val_dataloader.pin_memory)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True)

    num_devices = trainer_params.get('devices', 1)
    if isinstance(num_devices, int) and num_devices > 1:
        # IDOL (and similar agents) may skip submodules per step (warmup, heads, branches).
        # Default DDP assumes every parameter contributes to loss; unused params need this flag.
        trainer_params['strategy'] = "ddp_find_unused_parameters_true"
        trainer_params['sync_batchnorm'] = True
        logger.info(
            f"Multi-GPU DDP enabled: {num_devices} GPUs, sync_batchnorm=True, "
            "strategy=ddp_find_unused_parameters_true"
        )
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    version_coord_file = Path(cfg.output_dir) / ".ddp_logger_version"

    # A fixed logger version avoids a filesystem
    # race when external DDP ranks initialize before rank 0 writes its version.
    configured_version = cfg.get('logger_version', None)
    if configured_version is not None:
        tb_logger = TensorBoardLogger(
            save_dir=cfg.output_dir, name="lightning_logs",
            version=int(configured_version), default_hp_metric=False,
        )
        if local_rank == 0:
            Path(tb_logger.log_dir).mkdir(parents=True, exist_ok=True)
            version_coord_file.write_text(str(tb_logger.version))
    elif local_rank == 0:
        tb_logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="lightning_logs",
            default_hp_metric=False,
        )
        Path(tb_logger.log_dir).mkdir(parents=True, exist_ok=True)
        version_coord_file.write_text(str(tb_logger.version))
    else:
        version = int(version_coord_file.read_text().strip())
        tb_logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="lightning_logs",
            version=version,
            default_hp_metric=False,
        )

    checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    if local_rank == 0:
        log_dir = Path(tb_logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)

    callbacks = agent.get_training_callbacks()
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            callback.dirpath = str(checkpoint_dir)
    
    trainer = pl.Trainer(
                **trainer_params, 
                callbacks=callbacks,
                logger=tb_logger,
                )

    logger.info("Starting Training")
    ckpt_path = cfg.get("resume_trainer_ckpt_path", None)
    if ckpt_path:
        logger.info(f"Resuming full Lightning trainer state from: {ckpt_path}")

    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )

if __name__ == "__main__":
    main()
