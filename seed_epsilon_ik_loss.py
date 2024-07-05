import torch
import torch.nn as nn

from affine_loss import AffineLoss
from typing import Tuple
from fk import FK


class SeedEpsilonIKLoss(nn.Module):

    def __init__(self, device):
        super(SeedEpsilonIKLoss, self).__init__()
        self.affine_loss = AffineLoss(alpha=1.0, beta=1.0)
        self.device = device
        self.fk = FK(device)

    def forward(
            self,
            pred_pose: Tuple[torch.Tensor, torch.Tensor],
            target_pose: Tuple[torch.Tensor, torch.Tensor],
            pred_joints,
            seed_joints: torch.Tensor):
        d1 = self.affine_loss.loss_fn(pred_pose, target_pose)
        d2 = self.affine_loss.loss_fn(self.fk(pred_joints), self.fk(seed_joints))
        return torch.mean(d1 / d2)
