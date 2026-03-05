"""Modified Pri4R head with configurable point encoder for ablation studies.

Quick switch example:
    PRI4R_HEAD_CONFIG = {
        "hidden_size": 4096,
        "action_horizon": 8,
        "point_encoder_type": "mlp",  # "mlp" | "pointnet"
        "num_points": 1024,
    }

This head is training-only auxiliary supervision:
- Training: use this module to predict future point displacements.
- Inference: simply do not instantiate/call this head (zero inference overhead).
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.backbones.point.pointmlp import (
    ConvBNReLU1D,
    LocalGrouper,
    PosExtraction,
    PreExtraction,
    index_points,
    square_distance,
)
from prismatic.models.backbones.point.pointnet import STN3d


PRI4R_HEAD_CONFIG: Dict[str, object] = {
    "hidden_size": 4096,
    "action_horizon": 8,
    "point_encoder_type": "mlp",  # switch to "pointnet" for ablation
    "num_points": 1024,
    "fusion_hidden_size": 512,
}


class PointMLPEncoderAdapter(nn.Module):
    """PointMLP encoder adapter producing per-point features.

    This adapter reuses official PointMLP encoder components and returns features at
    the original input point count through 3-NN interpolation.

    Input:
        point_cloud: (B, Np, 3)
    Output:
        point_features: (B, Np, out_dim)
    """

    def __init__(
        self,
        out_dim: int,
        num_points: int = 1024,
        embed_dim: int = 64,
        groups: int = 1,
        res_expansion: float = 1.0,
        activation: str = "relu",
        bias: bool = False,
        use_xyz: bool = False,
        normalize: str = "anchor",
        dim_expansion: Optional[list[int]] = None,
        pre_blocks: Optional[list[int]] = None,
        pos_blocks: Optional[list[int]] = None,
        k_neighbors: Optional[list[int]] = None,
        reducers: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        dim_expansion = dim_expansion or [2, 2, 2, 2]
        pre_blocks = pre_blocks or [2, 2, 2, 2]
        pos_blocks = pos_blocks or [2, 2, 2, 2]
        k_neighbors = k_neighbors or [24, 24, 24, 24]
        reducers = reducers or [2, 2, 2, 2]

        if not (
            len(dim_expansion)
            == len(pre_blocks)
            == len(pos_blocks)
            == len(k_neighbors)
            == len(reducers)
        ):
            raise ValueError("PointMLP stage configs must have equal lengths.")

        self.num_points = num_points
        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = num_points
        for stage in range(self.stages):
            out_channel = last_channel * dim_expansion[stage]
            anchor_points = anchor_points // reducers[stage]

            self.local_grouper_list.append(
                LocalGrouper(
                    channel=last_channel,
                    groups=anchor_points,
                    kneighbors=k_neighbors[stage],
                    use_xyz=use_xyz,
                    normalize=normalize,
                )
            )
            self.pre_blocks_list.append(
                PreExtraction(
                    channels=last_channel,
                    out_channels=out_channel,
                    blocks=pre_blocks[stage],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                    use_xyz=use_xyz,
                )
            )
            self.pos_blocks_list.append(
                PosExtraction(
                    channels=out_channel,
                    blocks=pos_blocks[stage],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
            last_channel = out_channel

        self.out_proj = nn.Sequential(
            nn.Conv1d(last_channel, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    @staticmethod
    def _three_nn_interpolate(
        source_xyz: torch.Tensor,
        source_features: torch.Tensor,
        target_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """3-NN inverse-distance interpolation from source points to target points.

        Args:
            source_xyz: (B, S, 3)
            source_features: (B, C, S)
            target_xyz: (B, N, 3)
        Returns:
            interpolated: (B, C, N)
        """
        distances = square_distance(target_xyz, source_xyz)  # (B, N, S)
        distances, neighbor_idx = distances.sort(dim=-1)
        distances = distances[:, :, :3]
        neighbor_idx = neighbor_idx[:, :, :3]  # (B, N, 3)

        inv_distance = 1.0 / (distances + 1e-8)
        norm = inv_distance.sum(dim=2, keepdim=True)
        weights = inv_distance / norm  # (B, N, 3)

        source_features_bsc = source_features.transpose(1, 2)  # (B, S, C)
        interpolated = torch.sum(index_points(source_features_bsc, neighbor_idx) * weights.unsqueeze(-1), dim=2)
        return interpolated.transpose(1, 2)  # (B, C, N)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(f"`point_cloud` must be (B, Np, 3), got {tuple(point_cloud.shape)}")

        xyz = point_cloud  # (B, Np, 3)
        features = self.embedding(point_cloud.transpose(1, 2))  # (B, C0, Np)

        for stage in range(self.stages):
            xyz, grouped_features = self.local_grouper_list[stage](xyz, features.transpose(1, 2))
            features = self.pre_blocks_list[stage](grouped_features)
            features = self.pos_blocks_list[stage](features)  # (B, C_stage, Ns)

        features = self._three_nn_interpolate(source_xyz=xyz, source_features=features, target_xyz=point_cloud)
        features = self.out_proj(features)  # (B, out_dim, Np)
        return features.transpose(1, 2)  # (B, Np, out_dim)


class PointNetEncoderAdapter(nn.Module):
    """PointNet encoder adapter producing per-point features.

    Reuses PointNet's spatial transform and pointwise conv stack, then projects to out_dim.

    Input:
        point_cloud: (B, Np, 3)
    Output:
        point_features: (B, Np, out_dim)
    """

    def __init__(self, out_dim: int, use_feature_transform: bool = False) -> None:
        super().__init__()
        self.use_feature_transform = use_feature_transform

        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.out_proj = nn.Sequential(
            nn.Conv1d(1024, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(f"`point_cloud` must be (B, Np, 3), got {tuple(point_cloud.shape)}")

        points_bcn = point_cloud.transpose(1, 2).contiguous()  # (B, 3, Np)

        transform = self.stn(points_bcn)  # (B, 3, 3)
        aligned = torch.bmm(point_cloud, transform).transpose(1, 2).contiguous()  # (B, 3, Np)

        features = F.relu(self.bn1(self.conv1(aligned)))
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.bn3(self.conv3(features))  # (B, 1024, Np)

        features = self.out_proj(features)  # (B, out_dim, Np)
        return features.transpose(1, 2)  # (B, Np, out_dim)


class ModifiedPri4RHead(nn.Module):
    """Pri4R-style privileged head conditioned on NON-action VL tokens.

    Key difference from original Pri4R:
    - Condition uses non-action visual-language tokens (image/text hidden states)
      instead of action-query token embeddings.

    Point encoder is modular via `point_encoder_type` for ablation.
    """

    def __init__(
        self,
        hidden_size: int,
        action_horizon: int,
        point_encoder_type: Literal["mlp", "pointnet"] = "mlp",
        num_points: int = 1024,
        fusion_hidden_size: int = 512,
        ignore_index: int = -100,
        point_encoder_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_horizon = action_horizon
        self.ignore_index = ignore_index
        self.point_encoder_type = point_encoder_type
        self.num_points = num_points

        # 1) Build non-action token conditioning branch z'_t: (B, H, D)
        self.global_condition_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(1, action_horizon, hidden_size))
        self.temporal_refiner = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # 2) Build configurable point encoder e_t: (B, Np, D)
        point_encoder_kwargs = point_encoder_kwargs or {}
        self.point_encoder = self._build_point_encoder(
            point_encoder_type=point_encoder_type,
            out_dim=hidden_size,
            num_points=num_points,
            point_encoder_kwargs=point_encoder_kwargs,
        )

        # 3) Fusion head: concat([z'_t, e_t]) -> displacement (dx,dy,dz)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, 3),
        )

    @staticmethod
    def _build_point_encoder(
        point_encoder_type: str,
        out_dim: int,
        num_points: int,
        point_encoder_kwargs: Dict[str, object],
    ) -> nn.Module:
        """Factory for point encoders; easy to extend for future backbones."""
        if point_encoder_type == "mlp":
            return PointMLPEncoderAdapter(out_dim=out_dim, num_points=num_points, **point_encoder_kwargs)
        if point_encoder_type == "pointnet":
            return PointNetEncoderAdapter(out_dim=out_dim, **point_encoder_kwargs)
        raise ValueError(
            f"Unsupported `point_encoder_type`={point_encoder_type}. "
            "Expected one of ['mlp', 'pointnet']."
        )

    @staticmethod
    def build_action_mask_from_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """Build action-token mask where True indicates action-token positions."""
        if labels.ndim != 2:
            raise ValueError(f"`labels` must have shape (B, S), got {tuple(labels.shape)}")
        return labels.ne(ignore_index)

    @staticmethod
    def build_action_mask_from_token_ids(token_ids: torch.Tensor, action_token_begin_idx: int) -> torch.Tensor:
        """Build action-token mask from token ids via tokenizer boundary."""
        if token_ids.ndim != 2:
            raise ValueError(f"`token_ids` must have shape (B, S), got {tuple(token_ids.shape)}")
        return token_ids.gt(action_token_begin_idx)

    def _pool_non_action_tokens(
        self,
        hidden_states: torch.Tensor,
        action_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool non-action VL tokens from backbone hidden states.

        Args:
            hidden_states: (B, S, D)
            action_mask: (B, S), True for action tokens
            attention_mask: (B, S), optional valid-token mask
        Returns:
            pooled_non_action: (B, D)
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"`hidden_states` must be (B,S,D), got {tuple(hidden_states.shape)}")
        if action_mask.ndim != 2 or hidden_states.shape[:2] != action_mask.shape:
            raise ValueError(
                "`action_mask` must be (B,S) and aligned with hidden states, "
                f"got hidden_states={tuple(hidden_states.shape)}, action_mask={tuple(action_mask.shape)}"
            )

        non_action_mask = ~action_mask
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != action_mask.shape:
                raise ValueError(
                    f"`attention_mask` must be {tuple(action_mask.shape)}, got {tuple(attention_mask.shape)}"
                )
            non_action_mask = non_action_mask & attention_mask.bool()

        valid_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(non_action_mask)
        empty_rows = non_action_mask.sum(dim=1) == 0
        if empty_rows.any():
            non_action_mask = torch.where(empty_rows[:, None], valid_mask, non_action_mask)

        weights = non_action_mask.to(hidden_states.dtype).unsqueeze(-1)  # (B,S,1)
        denom = weights.sum(dim=1).clamp(min=1.0)  # (B,1)
        pooled_non_action = (hidden_states * weights).sum(dim=1) / denom  # (B,D)
        return pooled_non_action

    def forward(
        self,
        hidden_states: torch.Tensor,
        point_cloud: torch.Tensor,
        action_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Predict future point displacement trajectories.

        Args:
            hidden_states: (B, S, D) last hidden states from VLA backbone.
            point_cloud: (B, Np, 3), default Np=1024.
            action_mask: (B, S), True at action-token positions.
            attention_mask: (B, S), optional valid-token mask.
        Returns:
            delta_p_hat: (B, H, Np, 3)
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"`hidden_states` must be (B,S,D), got {tuple(hidden_states.shape)}")
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(f"`point_cloud` must be (B,Np,3), got {tuple(point_cloud.shape)}")

        batch_size, _, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(f"Hidden dim mismatch: got {hidden_dim}, expected {self.hidden_size}")
        if point_cloud.shape[1] != self.num_points:
            raise ValueError(f"Point count mismatch: got {point_cloud.shape[1]}, expected {self.num_points}")

        # 1) Non-action VL token conditioning -> z_t' with horizon alignment
        # pooled_non_action: (B, D)
        pooled_non_action = self._pool_non_action_tokens(hidden_states, action_mask, attention_mask)
        # base_condition: (B,D) -> (B,1,D) -> (B,H,D)
        base_condition = self.global_condition_projector(pooled_non_action).unsqueeze(1).expand(
            -1, self.action_horizon, -1
        )
        # z_t_prime: (B, H, D)
        z_t_prime = self.temporal_refiner(base_condition + self.horizon_embedding)

        # 2) Point encoder output e_t: (B, Np, D)
        e_t = self.point_encoder(point_cloud)

        # 3) Broadcast and fuse
        # z_broadcast: (B,H,1,D) -> (B,H,Np,D)
        z_broadcast = z_t_prime.unsqueeze(2).expand(-1, -1, e_t.shape[1], -1)
        # e_broadcast: (B,1,Np,D) -> (B,H,Np,D)
        e_broadcast = e_t.unsqueeze(1).expand(-1, self.action_horizon, -1, -1)

        # fused: (B, H, Np, 2D)
        fused = torch.cat([z_broadcast, e_broadcast], dim=-1)

        # delta_p_hat: (B, H, Np, 3)
        delta_p_hat = self.fusion_mlp(fused)

        if return_intermediates:
            return {
                "delta_p_hat": delta_p_hat,
                "z_t_prime": z_t_prime,
                "e_t": e_t,
                "point_encoder_type": torch.tensor(0 if self.point_encoder_type == "mlp" else 1),
                "batch_size": torch.tensor(batch_size),
            }
        return delta_p_hat


def pri4r_displacement_l1_loss(
    pred_displacements: torch.Tensor,
    target_displacements: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 loss on displacement trajectories.

    Args:
        pred_displacements: (B, H, Np, 3)
        target_displacements: (B, H, Np, 3)
        reduction: reduction mode for torch.nn.functional.l1_loss
    """
    if pred_displacements.shape != target_displacements.shape:
        raise ValueError(
            "Shape mismatch in displacement loss: "
            f"pred={tuple(pred_displacements.shape)}, target={tuple(target_displacements.shape)}"
        )
    if pred_displacements.ndim != 4 or pred_displacements.shape[-1] != 3:
        raise ValueError(
            "Displacements must have shape (B, H, Np, 3), "
            f"got {tuple(pred_displacements.shape)}"
        )
    return F.l1_loss(pred_displacements, target_displacements, reduction=reduction)
