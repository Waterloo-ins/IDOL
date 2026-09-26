from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import timm
import time

from navsim.common.enums import StateSE2Index

import torchvision.models as models
import torch.nn.functional as F

import os
from datetime import datetime
from navsim.agents.IDOL.IDOL_targets import BoundingBox2DIndex
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from typing import Any, List, Dict, Union


class InverseDynamicsModel(nn.Module):
    """
    Inverse Dynamics Model (IDM)

    Dual-output IDM: produces both per-position spatial dynamics and a pooled
    global dynamics vector from adjacent BEV frames.

    Inputs:
        bev_current: [B*T, H*W, C]
        bev_next:    [B*T, H*W, C]

    Outputs:
        spatial_dynamics: [B*T, H*W, C]  — per-BEV-token dynamics (WHERE changes happen)
        global_dynamics:  [B*T, C]        — pooled dynamics (WHAT changed overall)
    """
    def __init__(
        self, 
        hidden_dim: int = 256, 
        dropout: float = 0.1
    ):
        super(InverseDynamicsModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.mlp_encoder_layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.mlp_encoder_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.mlp_encoder_layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.spatial_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, bev_current: torch.Tensor, bev_next: torch.Tensor):
        """
        Returns:
            spatial_dynamics: [B*T, H*W, C]
            global_dynamics:  [B*T, C]
        """
        fused = torch.cat([bev_current, bev_next], dim=-1)  # [B*T, H*W, 2C]
        fused = self.feature_fusion(fused)                   # [B*T, H*W, C]
        
        encoded = self.mlp_encoder_layer1(fused) + fused       # [B*T, H*W, C]
        encoded = self.mlp_encoder_layer2(encoded) + encoded   # [B*T, H*W, C]
        encoded = self.mlp_encoder_layer3(encoded) + encoded   # [B*T, H*W, C]
        
        spatial_dynamics = self.spatial_proj(encoded)          # [B*T, H*W, C]
        
        pooled = encoded.mean(dim=1)                           # [B*T, C]
        global_dynamics = self.output_head(pooled)             # [B*T, C]
        
        return spatial_dynamics, global_dynamics




class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.GELU(),
            nn.Linear(f_dim, f_dim),
            nn.GELU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out
class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34Backbone, self).__init__()
        # Load a pre-trained ResNet-34 model
        resnet = models.resnet34(pretrained=pretrained)
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # Extract features from the last convolutional layer
        res = self.backbone(x)
        return res            
       
class IDOLModel(nn.Module):
    def __init__(self, 
                 config,
                ):
        super().__init__()
        
        self.config = config

        # Define constants as variables
        STATUS_ENCODING_INPUT_DIM = 4 + 2 + 2
        hidden_dim = 256
        NUM_CLUSTERS = config.n_clusters if hasattr(config, 'n_clusters') else 256
        CLUSTER_CENTERS_FEATURE_DIM = 24

        TRANSFORMER_DIM_FEEDFORWARD = 512
        TRANSFORMER_NHEAD = 8
        TRANSFORMER_DROPOUT = 0.1
        TRANSFORMER_NUM_LAYERS = 2

        SCORE_HEAD_HIDDEN_DIM = 128
        SCORE_HEAD_OUTPUT_DIM = 1
        NUM_SCORE_HEADS = 5

        self._backbone = TransfuserBackbone(config)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)

        self._status_encoding = nn.Linear(STATUS_ENCODING_INPUT_DIM, config.tf_d_model)

        num_poses = config.trajectory_sampling.num_poses
        self.num_poses = num_poses
        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 64
        self.num_plan_queries = num_keyval
        self._keyval_embedding = nn.Embedding(
            num_keyval, config.tf_d_model
        )

        cluster_file = config.cluster_file_path
        self.register_buffer(
            'trajectory_anchors',
            torch.tensor(np.load(cluster_file)),
        )

        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(CLUSTER_CENTERS_FEATURE_DIM, SCORE_HEAD_HIDDEN_DIM),
            nn.LayerNorm(SCORE_HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer Encoder for trajectory_anchors_feat
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=TRANSFORMER_NHEAD, 
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, 
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.cluster_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=4,
            norm=nn.LayerNorm(hidden_dim),
        )
        
        # latent world model
        self.num_scenes = self.num_plan_queries + 1  # including the ego feat and action
        self.scene_position_embedding = nn.Embedding(self.num_scenes, hidden_dim)

        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        wm_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=TRANSFORMER_NHEAD, 
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, 
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.latent_world_model = nn.TransformerEncoder(
            wm_encoder_layer, num_layers=4,
            norm=nn.LayerNorm(hidden_dim),
        )

        # reward conv net
        self.num_fut_timestep = config.num_fut_timestep if hasattr(config, 'num_fut_timestep') else 4
        self.reward_conv_net = RewardConvNet(input_channels=(self.num_fut_timestep+1) * hidden_dim, conv1_out_channels=hidden_dim, conv2_out_channels=hidden_dim)
        self.reward_cat_head = nn.Sequential(
            nn.Linear((self.num_fut_timestep + 1 + 1) * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP head for scoring
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM // 2, SCORE_HEAD_OUTPUT_DIM),
        )
        self.sim_reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_HIDDEN_DIM // 2),
                nn.GELU(),
                nn.Linear(SCORE_HEAD_HIDDEN_DIM // 2, SCORE_HEAD_OUTPUT_DIM),
            ) for _ in range(NUM_SCORE_HEADS)
        ])

        # agent
        self.use_agent_loss = config.use_agent_loss if hasattr(config, 'use_agent_loss') else True
        if self.use_agent_loss:
            self.agent_query_embedding = nn.Embedding(config.num_bounding_boxes, hidden_dim)
            self.agent_head = AgentHead(
                num_agents=config.num_bounding_boxes,
                d_ffn=config.tf_d_ffn,
                d_model=config.tf_d_model,
            )

            agent_tf_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
                norm_first=True,
                activation='gelu',
            )

            self.agent_tf_decoder = nn.TransformerDecoder(
                agent_tf_decoder_layer, config.tf_num_layers,
                norm=nn.LayerNorm(config.tf_d_model),
            )

        # map
        self.use_map_loss = config.use_map_loss if hasattr(config, 'use_map_loss') else False
        if self.use_map_loss:
            self.bev_upsample_head = BEVUpsampleHead(config)
            self.bev_semantic_head = nn.Sequential(
                nn.Conv2d(
                    config.bev_features_channels,
                    config.bev_features_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    config.bev_features_channels,
                    config.num_bev_classes,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Upsample(
                    size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        self._bev_upscale = nn.Conv2d(config.tf_d_model, 512, kernel_size=1)
        
        # future agent & map
        self.num_sampled_trajs = config.num_sampled_trajs if hasattr(config, 'num_sampled_trajs') else 1
        self.new_scene_bev_feature_pos_embed = nn.Embedding(self.num_plan_queries, hidden_dim)

        # offset
        offset_tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )

        self.offset_tf_decoder = nn.TransformerDecoder(
            offset_tf_decoder_layer, config.tf_num_layers,
            norm=nn.LayerNorm(config.tf_d_model),
        )
        self.offset_head = TrajectoryOffsetHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        self.offset_score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM // 2, SCORE_HEAD_OUTPUT_DIM),
        )
        self.reward_weights = config.reward_weights if hasattr(config, 'reward_weights') else [0.1, 0.5, 0.5, 1.0]
        
        self.use_inverse_dynamics = config.use_inverse_dynamics if hasattr(config, 'use_inverse_dynamics') else True
        
        self.num_closed_loop_iters = config.num_closed_loop_iters if hasattr(config, 'num_closed_loop_iters') else 1

        self.use_idm_latent_corr = getattr(config, "use_idm_latent_corr", False)
        if self.use_idm_latent_corr:
            if not self.use_inverse_dynamics or self.num_closed_loop_iters < 1:
                raise ValueError("IDM latent correction requires active inverse dynamics")
            if self.num_fut_timestep != 1:
                raise ValueError("IDM latent correction currently requires num_fut_timestep=1")
            if num_poses < 2 or config.idm_latent_corr_weight <= 0:
                raise ValueError("IDM latent correction requires >=2 poses and a positive loss weight")
        
        if self.use_inverse_dynamics:
            self.inverse_dynamics_model = InverseDynamicsModel(
                hidden_dim=hidden_dim,
                dropout=TRANSFORMER_DROPOUT
            )
            
            # self.idm_traj_decoder = nn.Sequential(
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, num_poses * 3),
            # )
            
            # IDM spatial cross-attention: ego_feat queries the per-position
            # dynamics map to learn WHERE in BEV changes happened, then MLN
            # applies the global dynamics as a channel-wise modulation.
            self.idm_spatial_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
            )
            self.idm_spatial_attn_norm = nn.LayerNorm(hidden_dim)

            self.idm_fusion_mln = MLN(c_dim=hidden_dim, f_dim=hidden_dim)
            with torch.no_grad():
                self.idm_fusion_mln.gamma.weight.normal_(std=0.01)
                self.idm_fusion_mln.beta.weight.normal_(std=0.01)
            
            self.closed_loop_fusion_mln = MLN(c_dim=hidden_dim, f_dim=hidden_dim)
            with torch.no_grad():
                self.closed_loop_fusion_mln.gamma.weight.normal_(std=0.01)
                self.closed_loop_fusion_mln.beta.weight.normal_(std=0.01)

        # Created after existing modules to preserve their initialization order.
        # No query, anchor encoding, or trajectory-decoder input reaches this head.
        if self.use_idm_latent_corr:
            self.idm_motion_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3 * num_poses - 1),
            )

    def encode_traj_into_ego_feat(self, ego_status_feat: torch.Tensor, init_trajectory_anchor: torch.Tensor, batch_size: int):
        """
        Encode trajectory into ego feature by processing cluster centers and concatenating features.

        Args:
            ego_status_feat (torch.Tensor): Ego status features.
            init_trajectory_anchor (torch.Tensor): Initial cluster centers.
            batch_size (int): Batch size.

        Returns:
            ego_feat_fixed_anchor_IDOL (torch.Tensor): Concatenated ego and trajectory features.
            num_traj (int): Number of trajectories.
        """
        trajectory_anchors_feat, num_traj = self._get_trajectory_anchors_feat(init_trajectory_anchor, batch_size)
        ego_feat_encoded = self._concatenate_ego_and_traj_features(ego_status_feat, trajectory_anchors_feat)
        return ego_feat_encoded, num_traj

    def extract_trajectory_feature(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, Any]:
        results = {}

        camera_feature = features["camera_feature"]
        if getattr(self.config, "latent", False):
            lidar_feature = None
        else:
            lidar_feature = features["lidar_feature"]
        status_feature = features["status_feature"]

        batch_size = status_feature.shape[0]

        backbone_bev_feature, flatten_bev_feature = self._process_backbone_features(
            camera_feature, lidar_feature
        )

        ego_status_feat = self._get_ego_status_feature(status_feature)

        init_trajectory_anchor = self.trajectory_anchors.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        ego_feat, num_traj = self.encode_traj_into_ego_feat(ego_status_feat, init_trajectory_anchor, batch_size)

        if self.use_agent_loss:
            agents, agents_query = self._process_agent(batch_size, flatten_bev_feature)
            results.update(agents)
        if self.use_map_loss:
            bev_semantic_map, upsampled_bev_feature = self._process_map(flatten_bev_feature, batch_size)
            results["bev_semantic_map"] = bev_semantic_map

        trajectory_outputs = {
            "results": results,
            "batch_size": batch_size,
            "num_traj": num_traj,
            "flatten_bev_feature": flatten_bev_feature,
            "ego_feat": ego_feat,
        }
        return trajectory_outputs

    @staticmethod
    def _vis_detach_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu()

    def _infer_bev_hw(self, num_tokens: int = None):
        num_tokens = int(num_tokens or self.num_plan_queries)
        for h_name, w_name in (
            ("bev_token_height", "bev_token_width"),
            ("latent_bev_height", "latent_bev_width"),
            ("bev_h", "bev_w"),
        ):
            h_value = getattr(self.config, h_name, None)
            w_value = getattr(self.config, w_name, None)
            if h_value is not None and w_value is not None and int(h_value) * int(w_value) == num_tokens:
                return int(h_value), int(w_value)

        side = int(round(num_tokens ** 0.5))
        if side * side == num_tokens:
            return side, side

        h = int(np.floor(num_tokens ** 0.5))
        while h > 1 and num_tokens % h != 0:
            h -= 1
        return h, num_tokens // h

    def extract_reward_feature(
        self,
        trajectory_outputs,
        targets,
        return_vis: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """

        
          
          
          
          
          
        ==========================================================
        """
        results = trajectory_outputs["results"]
        batch_size = trajectory_outputs["batch_size"]
        num_traj = trajectory_outputs["num_traj"]
        flatten_bev_feature = trajectory_outputs["flatten_bev_feature"]  # [B, HW, C]
        ego_feat_init = trajectory_outputs["ego_feat"]  # [B, num_traj, 1, C]
        bev_h, bev_w = self._infer_bev_hw(flatten_bev_feature.shape[1])

        vis_dict = None
        if return_vis:
            vis_dict = {
                "bev_hw": torch.tensor([bev_h, bev_w], dtype=torch.long),
                "current_bev": self._vis_detach_cpu(flatten_bev_feature),
                "future_bev_states": [],
                "idm_spatial_maps": [],
                "idm_global_features": [],
                "idm_spatial_attention": [],
                "refinement_queries": [self._vis_detach_cpu(ego_feat_init)],
            }

        num_wm_steps = self.num_fut_timestep
        interval = 8 // num_wm_steps

        ego_feat_current = ego_feat_init  # [B, num_traj, 1, C]

        num_cl_iters = self.num_closed_loop_iters if self.use_inverse_dynamics else 1

        for cl_iter in range(num_cl_iters):
            if cl_iter > 0 and self.use_inverse_dynamics:
                ego_init_sq = ego_feat_init.squeeze(2)                  # [B, num_traj, C]
                ego_prev_refined_sq = ego_feat_current.squeeze(2)       # [B, num_traj, C]
                ego_feat_fused = self.closed_loop_fusion_mln(
                    ego_init_sq, ego_prev_refined_sq
                )  # [B, num_traj, C]
                ego_feat_current = ego_feat_fused.unsqueeze(2)          # [B, num_traj, 1, C]


            flatten_bev_feature_multi_trajs = flatten_bev_feature.unsqueeze(1).repeat(
                1, num_traj, 1, 1
            )

            flatten_bev_feature_multi_trajs = self._inject_cur_ego_into_bev(
                flatten_bev_feature_multi_trajs, ego_feat_current, num_traj, h=bev_h, w=bev_w
            )  # [B, num_traj, HW, C]

            ego_feat_wm = ego_feat_current.reshape(
                batch_size * num_traj, 1, -1
            )
            flatten_bev_feature_multi_trajs = flatten_bev_feature_multi_trajs.reshape(
                batch_size * num_traj, self.num_plan_queries, -1
            )

            fut_ego_feat = ego_feat_wm                                    # [B*num_traj, 1, C]
            ego_feat_list = [fut_ego_feat]
            fut_flatten_bev_feature_multi_trajs = flatten_bev_feature_multi_trajs  # [B*num_traj, HW, C]
            bev_feat_list = [fut_flatten_bev_feature_multi_trajs]

            for wm_step in range(num_wm_steps):
                fut_ego_feat, fut_flatten_bev_feature_multi_trajs = self._latent_world_model_processing(
                    fut_flatten_bev_feature_multi_trajs, fut_ego_feat, batch_size, num_traj
                )  # fut_ego_feat: [B*num_traj, 1, C], fut_bev: [B*num_traj, HW, C]
                fut_flatten_bev_feature_multi_trajs = self._inject_fut_ego_into_bev(
                    fut_flatten_bev_feature_multi_trajs,
                    fut_ego_feat,
                    num_traj,
                    fut_idx=(wm_step + 1) * interval,
                    h=bev_h,
                    w=bev_w,
                )  # [B*num_traj, HW, C]
                ego_feat_list.append(fut_ego_feat)                          # [B*num_traj, 1, C]
                bev_feat_list.append(fut_flatten_bev_feature_multi_trajs)   # [B*num_traj, HW, C]

            if return_vis:
                vis_dict["future_bev_states"].append(
                    [
                        self._vis_detach_cpu(
                            bev_feat.view(batch_size, num_traj, self.num_plan_queries, -1)
                        )
                        for bev_feat in bev_feat_list
                    ]
                )

            if self.use_inverse_dynamics:
                idm_outputs = self._inverse_dynamics_prediction(
                    bev_feat_list,
                    batch_size,
                    num_traj,
                    interval,
                    return_all=return_vis,
                )
                if return_vis:
                    spatial_idm, global_idm, all_spatial_idm, all_global_idm = idm_outputs
                    vis_dict["idm_spatial_maps"].append(self._vis_detach_cpu(all_spatial_idm))
                    vis_dict["idm_global_features"].append(self._vis_detach_cpu(all_global_idm))
                else:
                    spatial_idm, global_idm = idm_outputs
                if self.use_idm_latent_corr and targets is not None and cl_iter == num_cl_iters - 1:
                    results["idm_motion_residual"] = self.idm_motion_head(global_idm)
                # spatial: [B*num_traj, HW, C]  global: [B, num_traj, C]

                # query: ego_feat [B*T, 1, C],  key/value: spatial_idm [B*T, HW, C]
                ego_q = ego_feat_current.reshape(
                    batch_size * num_traj, 1, -1
                )  # [B*T, 1, C]
                attn_out, attn_weights = self.idm_spatial_cross_attn(
                    query=ego_q, key=spatial_idm, value=spatial_idm
                )  # [B*T, 1, C]
                if return_vis:
                    vis_dict["idm_spatial_attention"].append(
                        self._vis_detach_cpu(
                            attn_weights.view(batch_size, num_traj, 1, self.num_plan_queries)
                        )
                    )
                ego_enriched = self.idm_spatial_attn_norm(
                    ego_q + attn_out
                )  # [B*T, 1, C]  residual + LN
                ego_enriched = ego_enriched.view(
                    batch_size, num_traj, -1
                )  # [B, num_traj, C]

                ego_feat_refined = self.idm_fusion_mln(
                    ego_enriched, global_idm
                )  # [B, num_traj, C]

                ego_feat_current = ego_feat_refined.unsqueeze(2)            # [B, num_traj, 1, C]
                if return_vis:
                    vis_dict["refinement_queries"].append(self._vis_detach_cpu(ego_feat_current))


        fut_flatten_bev_feature_multi_trajs = bev_feat_list[-1]  # [B*num_traj, HW, C]

        reward_feature = self._compute_reward_feature(
            ego_feat_list,
            bev_feat_list,
            batch_size,
            num_traj,
            h=bev_h,
            w=bev_w,
        )
        results["reward_feature"] = reward_feature

        ego_feat_for_offset = ego_feat_current if self.use_inverse_dynamics else ego_feat_init
        offset_dict = self._predict_offset(ego_feat_for_offset, flatten_bev_feature)
        results.update(offset_dict)

        if targets is not None:
            sampled_fut_flatten_bev_feature_multi_trajs = self._sample_future_bev_feature(
                fut_flatten_bev_feature_multi_trajs,
                batch_size,
                num_traj,
                targets=targets
            )
            fut_bev_semantic_map = self._process_future_map(
                sampled_fut_flatten_bev_feature_multi_trajs,
                batch_size,
                h=bev_h,
                w=bev_w,
            )
            results["fut_bev_semantic_map"] = fut_bev_semantic_map

        if return_vis:
            results["vis_dict"] = vis_dict
        return results

    def process_trajectory_and_reward(
        self,
        features: Dict[str, torch.Tensor],
        targets=None,
        return_vis: bool = False,
    ) -> Dict[str, torch.Tensor]:
        trajectory_outputs = self.extract_trajectory_feature(features, targets)
        final_results = self.extract_reward_feature(
            trajectory_outputs,
            targets,
            return_vis=return_vis,
        )
        return final_results

    def _predict_offset(self, ego_feat: torch.Tensor, flatten_bev_feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the offset for the cluster centers.
        """
        offset_dict = {}
        ego_feat = ego_feat.squeeze(2)  # [batch_size, num_traj, C]
        ego_feat = self.offset_tf_decoder(ego_feat, flatten_bev_feature) # [batch_size, num_traj, C]
        trajectory_offset = self.offset_head(ego_feat)
        trajectory_offset_rewards = self.offset_score_head(ego_feat).squeeze(-1)  # [batch_size, num_traj]
        trajectory_offset_rewards = torch.softmax(trajectory_offset_rewards, dim=-1)
        offset_dict["trajectory_offset"] = trajectory_offset
        offset_dict["trajectory_offset_rewards"] = trajectory_offset_rewards
        return offset_dict

    def _process_backbone_features(self, camera_feature: torch.Tensor, lidar_feature: torch.Tensor):
        """
        Process the backbone network and extract BEV features.
        """
        _, backbone_bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        bev_feature = self._bev_downscale(backbone_bev_feature).flatten(-2, -1).permute(0, 2, 1)
        flatten_bev_feature = bev_feature + self._keyval_embedding.weight[None, :, :]
        return backbone_bev_feature, flatten_bev_feature

    def _get_ego_status_feature(self, status_feature: torch.Tensor) -> torch.Tensor:
        """
        Obtain the encoded ego vehicle status features.
        """
        status_encoding = self._status_encoding(status_feature)  # [batch_size, C]
        ego_status_feat = status_encoding[:, None, :]  # [batch_size, 1, C]
        return ego_status_feat

    def _get_trajectory_anchors_feat(self, trajectory_anchors, batch_size: int):
        """
        Get the features of the cluster centers and expand them to the batch size.
        """
        num_traj = trajectory_anchors.shape[1]
        device = trajectory_anchors.device
        init_traj = trajectory_anchors.reshape(batch_size, num_traj, -1).to(device)  # [bz, num_traj, D]
        trajectory_anchors_feat = self.mlp_planning_vb(init_traj)  # [bz, num_traj, hidden_dim]
        trajectory_anchors_feat = self.cluster_encoder(trajectory_anchors_feat)  # [bz, num_traj, encoded_dim]
        return trajectory_anchors_feat, num_traj

    def _concatenate_ego_and_traj_features(self, ego_status_feat: torch.Tensor, trajectory_anchors_feat: torch.Tensor) -> torch.Tensor:
        """
        Concatenate ego features with trajectory features and encode.
        """
        # Repeat ego_status_feat to match the number of trajectories
        ego_status_feat = ego_status_feat.repeat(1, trajectory_anchors_feat.shape[1], 1)  # [batch_size, num_traj, C]
        # Concatenate
        ego_feat = torch.cat([ego_status_feat, trajectory_anchors_feat], dim=-1)  # [batch_size, num_traj, C + encoded_dim]
        # Encode
        ego_feat = self.encode_ego_feat_mlp(ego_feat)  # [batch_size, num_traj, C']
        ego_feat = ego_feat.unsqueeze(-2)  # [batch_size, num_traj, 1, C']
        return ego_feat

    def _inject_cur_ego_into_bev(
        self,
        scene_bev_feature: torch.Tensor,
        ego_feat: torch.Tensor,
        num_traj: int,
        h=None,
        w=None,
    ) -> torch.Tensor:
        """
        Inject the ego feature into the BEV map.
        """
        if h is None or w is None:
            h, w = self._infer_bev_hw(scene_bev_feature.shape[2])
        bz = scene_bev_feature.shape[0]
        scene_bev_feature = scene_bev_feature.permute(0, 1, 3, 2).reshape(bz*num_traj, -1, h, w)  # [batch_size, num_traj, C, H*W]
        ego_feat = ego_feat.squeeze(2).reshape(bz*num_traj, -1)  # [batch_size*num_traj, C]
         # [batch_size*num_traj, 2]
        coors = torch.zeros(bz*num_traj, 2).to(ego_feat.device)
        scene_bev_feature = self.inject_ego_feat_to_bev_map(scene_bev_feature, ego_feat, coors, H=h, W=w)
        scene_bev_feature = scene_bev_feature.view(bz, num_traj, -1, h * w)  # [batch_size, num_traj, C, H*W]
        scene_bev_feature = scene_bev_feature.permute(0, 1, 3, 2)  # [batch_size, num_traj, H*W, C]
        return scene_bev_feature
    
    def _inject_fut_ego_into_bev(
        self,
        scene_bev_feature: torch.Tensor,
        ego_feat: torch.Tensor,
        num_traj: int,
        fut_idx=8,
        h=None,
        w=None,
    ) -> torch.Tensor:
        """
        Inject the ego feature into the BEV map.
        """
        if h is None or w is None:
            h, w = self._infer_bev_hw(scene_bev_feature.shape[1])
        bz = int(scene_bev_feature.shape[0] // num_traj)
        scene_bev_feature = scene_bev_feature.permute(0, 2, 1).reshape(bz*num_traj, -1, h, w)  # [batch_size, num_traj, C, H*W]
        ego_feat = ego_feat.squeeze(2).reshape(bz*num_traj, -1)  # [batch_size*num_traj, C]
         # [batch_size*num_traj, 2]
        # coors = torch.zeros(bz*num_traj, 2).to(ego_feat.device)
        coors = self.trajectory_anchors[:, fut_idx-1, :2].to(ego_feat.device).unsqueeze(0).repeat(bz, 1, 1).reshape(bz*num_traj, -1)

        scene_bev_feature = self.inject_ego_feat_to_bev_map(scene_bev_feature, ego_feat, coors, H=h, W=w)
        scene_bev_feature = scene_bev_feature.view(bz*num_traj, -1, h * w)  # [batch_size, num_traj, C, H*W]
        scene_bev_feature = scene_bev_feature.permute(0, 2, 1)  # [batch_size*num_traj, H*W, C]
        return scene_bev_feature

    def _latent_world_model_processing(self, flatten_bev_feature_multi_trajs: torch.Tensor, ego_feat: torch.Tensor, batch_size: int, num_traj: int) -> torch.Tensor:
        """
        Process features through the latent world model.
        """
        # Concatenate ego_feat and flatten_bev_feature_multi_trajs
        # Assume concatenation along the feature dimension
        scene_feature = torch.cat([ego_feat, flatten_bev_feature_multi_trajs], dim=1)  # [batch_size, num_traj, 1 + 32, C]
        
        # Add positional embedding
        scene_position_embedding = self.scene_position_embedding.weight[None,  :, :].repeat(batch_size * num_traj, 1, 1)  # [batch_size * num_traj, 1 + 32, embedding_dim]

        scene_feature = scene_feature + scene_position_embedding
        
        # Reshape to fit the latent world model
        fut_scene_feature = self.latent_world_model(scene_feature)  # [batch_size*num_traj, C_new, H'*W']
        fut_ego_feat = fut_scene_feature[:, 0:1]
        fut_flatten_bev_feature_multi_trajs = fut_scene_feature[:, 1:]
        return fut_ego_feat, fut_flatten_bev_feature_multi_trajs

    def _compute_reward_feature(
        self,
        fut_ego_feat_list,
        fut_flatten_bev_feature_multi_trajs_list,
        batch_size,
        num_traj,
        h=None,
        w=None,
    ) -> torch.Tensor:
        """
        Compute the scoring features.
        """
        if h is None or w is None:
            h, w = self._infer_bev_hw(fut_flatten_bev_feature_multi_trajs_list[0].shape[1])
        bev_feat_list = []
        for bev_feat in  fut_flatten_bev_feature_multi_trajs_list:
            bev_feat = bev_feat.view(batch_size * num_traj, h, w, -1) 
            bev_feat_list.append(bev_feat)
        all_bev_feature = torch.cat(bev_feat_list, dim=-1)  # [batch_size*num_traj, H, W, C1 + C2]
        all_bev_feature = all_bev_feature.permute(0, 3, 1, 2)  # [batch_size*num_traj, C1 + C2, H, W]
        
        # Apply convolution network
        reward_conv_output = self.reward_conv_net(all_bev_feature).squeeze(-1).permute(0, 2, 1)  # [batch_size*num_traj, 1, C_conv]
        
        # Prepare scoring features
        cat_reward_feature = torch.cat(fut_ego_feat_list + [reward_conv_output], dim=1)  # [batch_size*num_traj, 3, C_total]
        cat_reward_feature = cat_reward_feature.reshape(batch_size * num_traj, -1)  # [batch_size*num_traj, 3*C_total]
        
        # Apply scoring head
        reward_feature = self.reward_cat_head(cat_reward_feature)  # [batch_size*num_traj, 256]
        reward_feature = reward_feature.view(batch_size, num_traj, -1)  # [batch_size, num_traj, 256]
        return reward_feature
    
    def _inverse_dynamics_prediction(
        self, 
        bev_feat_list: List[torch.Tensor], 
        batch_size: int, 
        num_traj: int,
        interval: int = 2,
        return_all: bool = False,
    ):
        num_frames = len(bev_feat_list)
        all_spatial = []
        all_global = []
        
        for i in range(num_frames - 1):
            bev_current = bev_feat_list[i]   # [B*num_traj, H*W, C]
            bev_next = bev_feat_list[i + 1]  # [B*num_traj, H*W, C]
            spatial_feat, global_feat = self.inverse_dynamics_model(bev_current, bev_next)
            all_spatial.append(spatial_feat)  # [B*num_traj, H*W, C]
            all_global.append(global_feat)   # [B*num_traj, C]
        
        all_spatial_tensor = torch.stack(all_spatial, dim=0)
        all_global_tensor = torch.stack(all_global, dim=0)

        if len(all_spatial) == 1:
            spatial_idm = all_spatial_tensor[0]
            global_idm = all_global_tensor[0]
        else:
            spatial_idm = all_spatial_tensor.mean(dim=0)
            global_idm = all_global_tensor.mean(dim=0)
        
        global_idm = global_idm.view(batch_size, num_traj, -1)  # [B, num_traj, C]

        if return_all:
            all_spatial_tensor = all_spatial_tensor.view(
                num_frames - 1,
                batch_size,
                num_traj,
                self.num_plan_queries,
                -1,
            )
            all_global_tensor = all_global_tensor.view(
                num_frames - 1,
                batch_size,
                num_traj,
                -1,
            )
            return spatial_idm, global_idm, all_spatial_tensor, all_global_tensor
        
        return spatial_idm, global_idm

    def _process_agent(self, batch_size: int, scene_bev_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process agent.
        """
        agent_query = self.agent_query_embedding.weight[None, :, :].repeat(batch_size, 1, 1)  # [batch_size, num_agents, hidden_dim]
        agent_query_out = self.agent_tf_decoder(agent_query, scene_bev_feature)  # [batch_size, num_agents, hidden_dim]
        agents = self.agent_head(agent_query_out)  # dict containing 'agent_states' and 'agent_labels'
        return agents, agent_query_out

    def _process_map(self, flatten_bev_feature: torch.Tensor, batch_size, h=None, w=None) -> torch.Tensor:
        """
        Process map.
        """
        # Adjust dimensions
        # flatten_bev_feature = flatten_bev_feature.mean(dim=1)  # [batch_size, H*W, C]
        if h is None or w is None:
            h, w = self._infer_bev_hw(flatten_bev_feature.shape[1])
        flatten_bev_feature = flatten_bev_feature.permute(0, 2, 1)  # [batch_size, C, H*W]
        bz, C, num_plan_queries = flatten_bev_feature.shape
        flatten_bev_feature = flatten_bev_feature.reshape(bz, C, h, w)  # [batch_size, C, H, W]
        flatten_bev_feature = self._bev_upscale(flatten_bev_feature)  # [batch_size, 2* C, H, W]

        upsampled_bev_feature = self.bev_upsample_head(flatten_bev_feature)  # [batch_size, C, H', W']
        bev_semantic_map = self.bev_semantic_head(upsampled_bev_feature)  # [batch_size, num_classes, H', W']
        return bev_semantic_map, upsampled_bev_feature

    def _sample_future_bev_feature(self, fut_scene_feature: torch.Tensor, batch_size: int, num_traj: int, targets) -> torch.Tensor:
        """
        Prepare future BEV features.
        """
        new_scene_bev_feature_pos_embed = self.new_scene_bev_feature_pos_embed.weight[None, :, :].repeat(batch_size * self.num_sampled_trajs, 1, 1)  # [batch_size*num_sampled_trajs, num_plan_queries, embedding_dim]
        new_scene_bev_feature = fut_scene_feature.view(batch_size, num_traj, self.num_plan_queries, -1)  # [batch_size, num_traj, num_plan_queries, C_new]
        B, T, H, W = new_scene_bev_feature.shape  # B=2, T=256, H=64, W=256
        sampled_trajs_index = targets['sampled_trajs_index']
        K = sampled_trajs_index.shape[1]  # K=3
        batch_indices = torch.arange(B, device=sampled_trajs_index.device).unsqueeze(1).repeat(1, K)  # shape: [B, K]
        selected_features = new_scene_bev_feature[batch_indices, sampled_trajs_index]  # shape: [B, K, H, W]
        new_scene_bev_feature = selected_features.view(batch_size * self.num_sampled_trajs, self.num_plan_queries, -1)  # [batch_size*num_sampled_trajs, num_plan_queries, C_new]
        new_scene_bev_feature_with_pos = new_scene_bev_feature + new_scene_bev_feature_pos_embed  # [batch_size*num_sampled_trajs, num_plan_queries, C_new]
        return new_scene_bev_feature_with_pos

    def _process_future_agents(self, new_scene_bev_feature_with_pos: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Process future agent features.
        """
        fut_agent_query = self.fut_agent_query_embedding.weight[None, :, :].repeat(batch_size * self.num_sampled_trajs, 1, 1)  # [batch_size*num_sampled_trajs, num_agents, hidden_dim]
        fut_agent_query_out = self.fut_agent_tf_decoder(fut_agent_query, new_scene_bev_feature_with_pos)  # [batch_size*num_sampled_trajs, num_agents, hidden_dim]
        fut_agents_dict = self.fut_agent_head(fut_agent_query_out)  # dict containing 'agent_states' and 'agent_labels'
        fut_agents_dict = {
            'fut_agent_states': fut_agents_dict.pop('agent_states'),
            'fut_agent_labels': fut_agents_dict.pop('agent_labels')
        }
        return fut_agents_dict

    def _process_future_map(
        self,
        new_scene_bev_feature_with_pos: torch.Tensor,
        batch_size: int,
        h=None,
        w=None,
    ) -> torch.Tensor:
        """
        Process future BEV semantic map.
        """
        # Adjust dimensions
        if h is None or w is None:
            h, w = self._infer_bev_hw(new_scene_bev_feature_with_pos.shape[1])
        fut_bev_feature = new_scene_bev_feature_with_pos.permute(0, 2, 1)  # [batch_size*num_sampled_trajs, C_new, num_plan_queries]
        fut_bev_feature = fut_bev_feature.reshape(batch_size * self.num_sampled_trajs, -1, h, w)  # [batch_size*num_sampled_trajs, C_new, 4, 8]
        fut_bev_feature = self._bev_upscale(fut_bev_feature)  # [batch_size*num_sampled_trajs, C_upscaled, H', W']
        upsampled_fut_bev_feature = self.bev_upsample_head(fut_bev_feature)  # [batch_size*num_sampled_trajs, C_upscaled, H'', W'']
        fut_bev_semantic_map = self.bev_semantic_head(upsampled_fut_bev_feature)  # [batch_size*num_sampled_trajs, num_classes, H'', W'']
        return fut_bev_semantic_map
    
    def inject_ego_feat_to_bev_map(self, bev_map, new_features, delta_x_y, H=8, W=8):
        """`
        Add a new feature vector in batch to the corresponding location in the BEV feature map, affecting the four pixels around each position.

        Parameters:
        - bev_map (torch.Tensor): BEV feature map with shape (B, C, H, W)
        - delta_x_y (torch.Tensor): x and y coordinates in the ego coordinate system, shape (B, 2)
        - new_features (torch.Tensor): Feature vectors to be added, shape (B, C)
        - H (int): Height (in pixels) of the BEV feature map
        - W (int): Width (in pixels) of the BEV feature map

        Returns:
        - updated_bev_map (torch.Tensor): Updated BEV feature map, shape (B, C, H, W)
        """
        B, C, H_map, W_map = bev_map.shape
        assert H_map == H and W_map == W, f"BEV map dimensions must be ({H}, {W}), but got ({H_map}, {W_map})"
        assert new_features.shape == (B, C), "new_features must have shape (B, C)"

        device = bev_map.device
        dtype = bev_map.dtype

        delta_x, delta_y = delta_x_y[:, 0], delta_x_y[:, 1]
        # Calculate the pixel-to-meter ratio
        pixel_per_meter_x = H / 32.0  # 32 meters covered by H pixels
        pixel_per_meter_y = W / 64.0  # 64 meters covered by W pixels

        # Convert ego coordinates to floating point pixel indices
        h_idx = delta_x * pixel_per_meter_x  # (B,)
        w_idx = delta_y * pixel_per_meter_y + (W / 2.0)  # Origin at (0, W/2)

        # Get four nearby integer pixel indices
        h0 = torch.floor(h_idx).long()  # (B,)
        w0 = torch.floor(w_idx).long()  # (B,)
        h1 = h0 + 1
        w1 = w0 + 1

        # Compute distance weights
        dh = h_idx - h0.float()  # (B,)
        dw = w_idx - w0.float()  # (B,)

        w00 = (1 - dh) * (1 - dw)  # (B,)
        w01 = (1 - dh) * dw
        w10 = dh * (1 - dw)
        w11 = dh * dw

        # Stack indices and weights for all four nearby pixels
        # Each point contributes to four positions
        h_indices = torch.stack([h0, h0, h1, h1], dim=1)  # (B, 4)
        w_indices = torch.stack([w0, w1, w0, w1], dim=1)  # (B, 4)
        weights = torch.stack([w00, w01, w10, w11], dim=1)  # (B, 4)

        # Create batch indices
        batch_indices = torch.arange(B, device=device).view(B, 1).repeat(1, 4)  # (B, 4)

        # Flatten contributions
        h_indices_flat = h_indices.reshape(-1)  # (B*4,)
        w_indices_flat = w_indices.reshape(-1)  # (B*4,)
        weights_flat = weights.reshape(-1)  # (B*4,)
        batch_indices_flat = batch_indices.reshape(-1)  # (B*4,)

        # Create validity mask to ensure indices are within range
        valid = (h_indices_flat >= 0) & (h_indices_flat < H) & (w_indices_flat >= 0) & (w_indices_flat < W)  # (B*4,)

        if not valid.any():
            print("No valid indices")
            return bev_map

        # Filter out invalid indices and weights
        h_indices_valid = h_indices_flat[valid]  # (M,)
        w_indices_valid = w_indices_flat[valid]  # (M,)
        weights_valid = weights_flat[valid].unsqueeze(1)  # (M, 1)
        batch_indices_valid = batch_indices_flat[valid]  # (M,)

        # Get corresponding feature vectors and weights
        # Repeat each new_feature four times corresponding to the four weights
        new_features_expanded = new_features[batch_indices_valid]  # (M, C)
        weighted_features = new_features_expanded * weights_valid  # (M, C)

        # Compute linear indices
        # linear_index = b * C * H * W + c * H * W + h * W + w
        # Compute indices for each channel using broadcasting
        c_indices = torch.arange(C, device=device).view(1, C).repeat(weighted_features.shape[0], 1)  # (M, C)
        linear_indices = (batch_indices_valid.unsqueeze(1) * C * H * W) + (c_indices * H * W) + (h_indices_valid.unsqueeze(1) * W) + w_indices_valid.unsqueeze(1)  # (M, C)
        linear_indices = linear_indices.reshape(-1)  # (M*C,)

        # Flatten weighted features to (M*C,)
        weighted_features_flat = weighted_features.reshape(-1)  # (M*C,)

        # Flatten bev_map to (B*C*H*W,)
        bev_map_flat = bev_map.reshape(-1)  # (B*C*H*W,)

        # Use index_add_ to accumulate weighted features at corresponding positions
        bev_map_flat.index_add_(0, linear_indices, weighted_features_flat)

        # Reshape the BEV feature map back to (B, C, H, W)
        updated_bev_map = bev_map_flat.view(B, C, H, W)

        return updated_bev_map
    
    def select_best_trajectory(self, final_rewards, trajectory_anchors, batch_size):
        best_trajectory_idx = torch.argmax(final_rewards, dim=-1)  # Shape: [batch_size]
        
        if trajectory_anchors.dim() == 3:
            # trajectory_anchors: [num_traj, num_poses, 3]
            poses = trajectory_anchors[best_trajectory_idx]  # Shape: [batch_size, num_poses, 3]
        else:
            # trajectory_anchors: [batch_size, num_traj, num_poses, 3]
            batch_indices = torch.arange(batch_size, device=trajectory_anchors.device)
            poses = trajectory_anchors[batch_indices, best_trajectory_idx]  # [batch_size, num_poses, 3]
        
        return poses

    def forward_test(self, features, targets=None) -> Dict[str, torch.Tensor]:
        self.is_eval = True
        encoder_results = self.process_trajectory_and_reward(features)
        cluster_feaure = encoder_results["reward_feature"]
        batch_size = cluster_feaure.shape[0]

        im_rewards = self.reward_head(cluster_feaure).squeeze(-1)
        im_rewards_softmax = torch.softmax(im_rewards, dim=-1)

        sim_rewards = [reward_head(cluster_feaure) for reward_head in self.sim_reward_heads]
        sim_rewards = [reward.sigmoid() for reward in sim_rewards]
        final_rewards = self.weighted_reward_calculation(im_rewards_softmax, sim_rewards)

        offset = encoder_results['trajectory_offset']
        trajectory_anchors_offset = self.trajectory_anchors.unsqueeze(0) + offset  # [B, num_traj, num_poses, 3]

        poses = self.select_best_trajectory(final_rewards, trajectory_anchors_offset, batch_size)

        results = {
            "trajectory": poses,
            "all_trajectory": trajectory_anchors_offset,
            "final_rewards": final_rewards,
            "im_rewards": im_rewards_softmax,
        }
        return results

    def forward_test_with_vis(self, features, targets=None):
        """
        Inference wrapper used only by explicit visualization entry points.

        The default forward_test path stays unchanged. This wrapper asks the
        closed-loop pipeline to detach CPU copies of IDM/BEV/query states, then
        decodes refinement-stage trajectories after the final prediction has
        already been computed.
        """
        self.is_eval = True
        encoder_results = self.process_trajectory_and_reward(features, return_vis=True)
        vis_dict = encoder_results.pop("vis_dict")
        cluster_feaure = encoder_results["reward_feature"]
        batch_size = cluster_feaure.shape[0]

        im_rewards = self.reward_head(cluster_feaure).squeeze(-1)
        im_rewards_softmax = torch.softmax(im_rewards, dim=-1)

        sim_rewards = [reward_head(cluster_feaure) for reward_head in self.sim_reward_heads]
        sim_rewards = [reward.sigmoid() for reward in sim_rewards]
        final_rewards = self.weighted_reward_calculation(im_rewards_softmax, sim_rewards)

        offset = encoder_results["trajectory_offset"]
        trajectory_anchors_offset = self.trajectory_anchors.unsqueeze(0) + offset
        selected_candidate_idx = torch.argmax(final_rewards, dim=-1)
        poses = self.select_best_trajectory(final_rewards, trajectory_anchors_offset, batch_size)

        results = {
            "trajectory": poses,
            "all_trajectory": trajectory_anchors_offset,
            "final_rewards": final_rewards,
            "im_rewards": im_rewards_softmax,
            "selected_candidate_idx": selected_candidate_idx,
        }

        refinement_trajectories = []
        current_bev = vis_dict["current_bev"].to(cluster_feaure.device)
        with torch.no_grad():
            for query_cpu in vis_dict.get("refinement_queries", []):
                query = query_cpu.to(cluster_feaure.device)
                offset_vis = self._predict_offset(query, current_bev)["trajectory_offset"]
                traj_vis = self.trajectory_anchors.unsqueeze(0) + offset_vis
                refinement_trajectories.append(self._vis_detach_cpu(traj_vis))

        vis_dict["refinement_trajectories"] = refinement_trajectories
        vis_dict["final_trajectory"] = self._vis_detach_cpu(poses)
        vis_dict["all_trajectory"] = self._vis_detach_cpu(trajectory_anchors_offset)
        vis_dict["final_rewards"] = self._vis_detach_cpu(final_rewards)
        vis_dict["im_rewards"] = self._vis_detach_cpu(im_rewards_softmax)
        vis_dict["selected_candidate_idx"] = self._vis_detach_cpu(selected_candidate_idx)

        return results, vis_dict

    def forward_train(self, features, targets=None) -> Dict[str, torch.Tensor]:
        self.is_eval = False
        result = {}
        encoder_results = self.process_trajectory_and_reward(features, targets)
        cluster_feaure = encoder_results.pop("reward_feature")

        im_rewards = self.reward_head(cluster_feaure).squeeze(-1)
        im_rewards_softmax = torch.softmax(im_rewards, dim=-1)
        result["im_rewards"] = im_rewards_softmax

        result["trajectory_anchors"] = self.trajectory_anchors

        sim_rewards = [sim_reward_head(cluster_feaure) for sim_reward_head in self.sim_reward_heads]
        sim_rewards = torch.cat(sim_rewards, dim=-1).permute(0, 2, 1).sigmoid()
        result["sim_rewards"] = sim_rewards

        result.update(encoder_results)

        return result

       
    def weighted_reward_calculation(self, im_rewards, sim_rewards) -> torch.Tensor:
        """
        Calculate the final reward for each trajectory based on the given weights.

        Args:
            im_rewards (torch.Tensor): Imitation rewards for each trajectory. Shape: [batch_size, num_traj]
            sim_rewards (List[torch.Tensor]): List of metric rewards for each trajectory. Each tensor shape: [batch_size, num_traj]
            w (List[float]): List of weights for combining the rewards.

        Returns:
            torch.Tensor: Final weighted reward for each trajectory. Shape: [batch_size, num_traj]
        """
        assert len(sim_rewards) == 5, "Expected 4 metric rewards: S_NC, S_DAC, S_TTC, S_EP, S_COMFORT"
        # Extract metric rewards
        w = self.reward_weights
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = sim_rewards
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = S_NC.squeeze(-1), S_DAC.squeeze(-1), S_EP.squeeze(-1), S_TTC.squeeze(-1), S_COMFORT.squeeze(-1)
        #self.metric_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort']
        # Calculate assembled cost based on the provided formula
        assembled_cost = (
            w[0] * torch.log(im_rewards) +
            w[1] * torch.log(S_NC) +
            w[2] * torch.log(S_DAC) +
            w[3] * torch.log(5 * S_TTC + 2 * S_COMFORT + 5 * S_EP)
        )
        return assembled_cost

class AgentHead(nn.Module):
    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:

        agent_states = self._mlp_states(agent_queries) # agent_states: torch.Size([32, 30, 5])
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}
    
class BEVUpsampleHead(nn.Module):
    def __init__(self, config, channel=64, c5_chs=512):
        super(BEVUpsampleHead, self).__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)

        # Initialize upsampling and convolution layers
        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor, mode="bilinear", align_corners=False
        )
        if config.lidar_min_x == 0.:
            self.upsample2 = nn.Upsample(
                    size=(
                        self.config.lidar_resolution_height // (2 * self.config.bev_down_sample_factor),
                        self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                    ),
                    mode="bilinear",
                    align_corners=False,
            )
        else:
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

        self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

        # Lateral connection
        self.c5_conv = nn.Conv2d(
            c5_chs, channel, (1, 1)
        )

    def forward(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

class RewardConvNet(nn.Module):
    def __init__(self, input_channels: int = 512, conv1_out_channels: int = 256, conv2_out_channels: int = 256):
        super(RewardConvNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, conv1_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)

        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)

        self.conv3 = nn.Conv2d(conv2_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv2_out_channels)

        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        identity = x
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.act(x + identity)
        x = self.pool(x)
        return x

class TrajectoryOffsetHead(nn.Module):
    def __init__(self, num_poses: int = 8, d_ffn: int=1024, d_model: int=256):
        super(TrajectoryOffsetHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.GELU(),
            nn.Linear(self._d_ffn, self._d_ffn // 2),
            nn.GELU(),
            nn.Linear(self._d_ffn // 2, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        bz, num_trajs, _ = object_queries.shape
        poses = self._mlp(object_queries).reshape(bz, -1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return poses
