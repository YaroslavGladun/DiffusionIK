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
            target_pose: Tuple[torch.Tensor, torch.Tensor],
            pred_delta,
            seed_joints: torch.Tensor):
        cartesian_distance = self.affine_loss.loss_fn(target_pose, self.fk(seed_joints + pred_delta))
        norm = self.affine_loss.loss_fn(target_pose, self.fk(seed_joints))
        cartesian_loss = torch.mean(cartesian_distance / norm)
        cartesian_loss_std = torch.std(cartesian_distance / norm)

        seed_loss = torch.norm(pred_delta, dim=-1) / norm
        seed_loss = torch.mean(seed_loss)

        # count where cartesian_loss >= 1
        cartesian_distance_norm = cartesian_distance / norm
        bad_predictions_ratio = torch.count_nonzero(cartesian_distance_norm >= 1) / cartesian_distance_norm.shape[0]

        return {"loss": 1e0 * cartesian_loss + 0 * seed_loss,
                "cartesian_loss": cartesian_loss,
                "cartesian_loss_std": cartesian_loss_std,
                "seed_loss": seed_loss,
                "bad_predictions_ratio": bad_predictions_ratio}
