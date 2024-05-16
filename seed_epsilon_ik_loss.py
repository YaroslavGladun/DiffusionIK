import torch
import torch.nn as nn

from affine_loss import AffineLoss
from typing import Tuple


class SeedEpsilonIKLoss(nn.Module):

    def __init__(self, device):
        super(SeedEpsilonIKLoss, self).__init__()
        self.affine_loss = AffineLoss(alpha=1.0, beta=1.0)
        self.device = device

    def forward(
            self,
            pred_pose: Tuple[torch.Tensor, torch.Tensor],
            target_pose: Tuple[torch.Tensor, torch.Tensor],
            pred_joints,
            seed_joints: torch.Tensor,
            epsilon: torch.Tensor
    ):
        affine_loss = self.affine_loss(pred_pose, target_pose)
        diff = pred_joints - seed_joints
        diff_cos = torch.cos(diff)
        diff_sin = torch.sin(diff)
        diff = torch.atan2(diff_sin, diff_cos)
        distance_to_seed = torch.sqrt(torch.sum(torch.square(diff), dim=-1))
        epsilon_loss = torch.mean(torch.abs(distance_to_seed - epsilon))
        return affine_loss + epsilon_loss
