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
            target_cosine_diff: torch.Tensor
    ):
        affine_loss = self.affine_loss(pred_pose, target_pose)
        pred_cosine_diff = torch.mean(torch.cos(pred_joints - seed_joints))
        seed_loss = torch.mean(torch.relu(pred_cosine_diff - target_cosine_diff))
        return affine_loss + seed_loss
