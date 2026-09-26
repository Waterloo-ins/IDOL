from typing import Any, List, Dict, Union
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision import transforms

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import timm, cv2
from navsim.agents.IDOL.IDOL_model import IDOLModel
from navsim.agents.IDOL.IDOL_loss import compute_IDOL_loss
from navsim.agents.IDOL.IDOL_targets import IDOLTargetBuilder
from navsim.agents.IDOL.IDOL_features import IDOLFeatureBuilder
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
import math
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class IDOLAgent(AbstractAgent):
    def __init__(
        self,
        config,
        trajectory_sampling: TrajectorySampling,
        lr: float,
        checkpoint_path: str = None,
        slice_indices=None,
        resume_from_checkpoint=False,
        use_wm=False,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._lr = lr
        self.max_epochs = config.max_epochs if hasattr(config, 'max_epochs') else 100
        self.min_lr = config.min_lr if hasattr(config, 'min_lr') else 1e-6

        self.IDOL_model = IDOLModel(config)

        self.slice_indices = (
            slice_indices
            if slice_indices is not None
            else [getattr(config, "sensor_frame_idx", 3)]
        )
        self.is_eval = False
        self.config = config

        if resume_from_checkpoint:
            self.initialize()

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._checkpoint_path is None:
            return

        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )["state_dict"]
        
        if "agent.IDOL_model.trajectory_anchors" in state_dict:
            del state_dict["agent.IDOL_model.trajectory_anchors"]

        cleaned_state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        model_state_dict = self.state_dict()

        shape_mismatched_keys = []
        filtered_state_dict = {}
        for key, value in cleaned_state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape != value.shape:
                shape_mismatched_keys.append(
                    (key, tuple(value.shape), tuple(model_state_dict[key].shape))
                )
                continue
            filtered_state_dict[key] = value

        incompatible = self.load_state_dict(filtered_state_dict, strict=False)

        missing_keys = [
            key for key in incompatible.missing_keys if key != "IDOL_model.trajectory_anchors"
        ]
        unexpected_keys = list(incompatible.unexpected_keys)

        if missing_keys or unexpected_keys:
            print(
                f"[IDOLAgent] Checkpoint load warnings: "
                f"{len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys."
            )
            if missing_keys:
                print(f"[IDOLAgent] Missing key sample: {missing_keys[:10]}")
            if unexpected_keys:
                print(f"[IDOLAgent] Unexpected key sample: {unexpected_keys[:10]}")

            backbone_related = [
                key
                for key in (missing_keys + unexpected_keys)
                if key.startswith("IDOL_model._backbone")
            ]
            if backbone_related:
                print(
                    "[IDOLAgent] Backbone-related checkpoint mismatch detected. "
                    "Please ensure train/eval use matching backbone and anchor settings."
                )

        if shape_mismatched_keys:
            print(
                f"[IDOLAgent] Ignored {len(shape_mismatched_keys)} checkpoint keys due to shape mismatch."
            )
            for key, ckpt_shape, model_shape in shape_mismatched_keys[:10]:
                print(
                    f"[IDOLAgent] Shape mismatch: {key} "
                    f"ckpt={ckpt_shape} model={model_shape}"
                )

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        if getattr(self.config, "latent", False):
            return SensorConfig(
                cam_f0=self.slice_indices,
                cam_l0=self.slice_indices,
                cam_l1=False,
                cam_l2=False,
                cam_r0=self.slice_indices,
                cam_r1=False,
                cam_r2=False,
                cam_b0=False,
                lidar_pc=False,
            )
        return SensorConfig.build_tfu_sensors(self.slice_indices) 

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            IDOLTargetBuilder(
                        trajectory_sampling=self._trajectory_sampling,
                        slice_indices=self.slice_indices,
                        sim_reward_dict_path=self.config.sim_reward_dict_path,
                        config=self.config,
                    ),
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [IDOLFeatureBuilder(self.slice_indices, self.config)]

    def forward(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        if not self.is_eval: #training
            return self.IDOL_model.forward_train(features, targets)
        else:
            return self.IDOL_model.forward_test(features)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        trajectory, _, _, _ = self.compute_trajectory_with_vis(agent_input)
        return trajectory

    def compute_trajectory_with_vis(self, agent_input: AgentInput):
        self.eval()
        self.is_eval = True

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        device = next(self.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.IDOL_model.forward_test(features)

        trajectory = Trajectory(predictions["trajectory"].squeeze(0).detach().cpu().numpy())
        all_traj_candidates = predictions["all_trajectory"].squeeze(0).detach().cpu().numpy()
        final_scores = predictions["final_rewards"].squeeze(0).detach().cpu().numpy()
        im_rewards = predictions["im_rewards"].squeeze(0).detach().cpu().numpy()

        return trajectory, all_traj_candidates, final_scores, im_rewards

    def compute_trajectory_with_idol_vis(self, agent_input: AgentInput):
        """
        Explicit visualization path for IDOL paper figures.

        Returns the normal selected trajectory plus the model predictions and a
        detached CPU vis_dict containing BEV rollouts, IDM maps, and refinement
        queries/trajectories. Default training/evaluation code never calls this.
        """
        self.eval()
        self.is_eval = True

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        device = next(self.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            predictions, vis_dict = self.IDOL_model.forward_test_with_vis(features)

        trajectory = Trajectory(predictions["trajectory"].squeeze(0).detach().cpu().numpy())
        return trajectory, predictions, vis_dict
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        current_epoch = getattr(self, 'current_epoch', 0)
        
        return compute_IDOL_loss(
            targets, 
            predictions, 
            self.config,
            current_epoch=current_epoch
        )
    
    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        use_coslr_opt = self.config.use_coslr_opt if hasattr(self.config, 'use_coslr_opt') else False
        if use_coslr_opt:
            return self.get_coslr_optimizers()
        else:
            return torch.optim.Adam(self.IDOL_model.parameters(), lr=self._lr)
    
    def get_training_callbacks(self) -> List[Any]:
        from pytorch_lightning.callbacks import ModelCheckpoint

        best_checkpoint = ModelCheckpoint(
            filename='best-{epoch:02d}-{step}',
            monitor='val/traj_offset_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
        
        periodic_checkpoint = ModelCheckpoint(
            filename='epoch={epoch:02d}-step={step}',
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            verbose=True,
        )
        
        return [best_checkpoint, periodic_checkpoint]
    
    def get_coslr_optimizers(self):
        optimizer_cfg = dict(type=self.config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self.config.weight_decay,
                            paramwise_cfg=self.config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self.config.scheduler_type,
                            milestones=self.config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)

        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self.IDOL_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self.IDOL_model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})

        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=self.min_lr,
            epochs=self.max_epochs,
            warmup_epochs=3,
        )

        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super(WarmupCosLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]
