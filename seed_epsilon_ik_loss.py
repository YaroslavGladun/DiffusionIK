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
        return self.get_affine_loss(pred_pose, target_pose)
        # mask = self.get_mask(pred_joints, seed_joints, target_cosine_diff)
        # if mask.sum() == 0:
        #     seed_loss = torch.tensor(0.0, device=self.device)
        # else:
        #     seed_loss = self.get_seed_loss(pred_joints[mask], seed_joints[mask], target_cosine_diff[mask])
        # affine_loss = self.affine_loss(pred_pose, target_pose)
        # return affine_loss + seed_loss

    def get_affine_loss(self, pred_pose: Tuple[torch.Tensor, torch.Tensor],
                        target_pose: Tuple[torch.Tensor, torch.Tensor]):
        return self.affine_loss(pred_pose, target_pose)

    def get_seed_loss(self, pred_joints, seed_joints: torch.Tensor, target_diff: torch.Tensor):
        pred_diff = torch.sqrt(torch.sum(torch.pow(pred_joints - seed_joints, 2), dim=-1, keepdim=True))
        seed_loss = torch.mean(torch.abs(pred_diff - target_diff))
        return seed_loss

    def get_mask(self, pred_joints, seed_joints: torch.Tensor, target_diff: torch.Tensor):
        pred_diff = torch.sqrt(torch.sum(torch.pow(pred_joints - seed_joints, 2), dim=-1, keepdim=True))
        mask = pred_diff > target_diff
        return mask.squeeze(dim=-1)
