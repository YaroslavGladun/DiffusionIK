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
        seed_loss = self.get_seed_loss(pred_joints, seed_joints, target_cosine_diff)
        return affine_loss + seed_loss

    def get_affine_loss(self, pred_pose: Tuple[torch.Tensor, torch.Tensor],
                        target_pose: Tuple[torch.Tensor, torch.Tensor]):
        return self.affine_loss(pred_pose, target_pose)

    def get_seed_loss(self, pred_joints, seed_joints: torch.Tensor, target_diff: torch.Tensor):
        pred_diff = torch.sqrt(torch.sum(torch.pow(pred_joints - seed_joints, 2), dim=-1, keepdim=True))
        seed_loss = torch.mean(torch.relu(pred_diff - target_diff))
        return seed_loss
