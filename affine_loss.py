from typing import Tuple

import torch
import torch.nn as nn


class AffineLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super(AffineLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: Tuple[torch.Tensor, torch.Tensor],
                target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        :param pred: R in shape (N, 3, 3), t in shape (N, 3, 1)
        :param target: R in shape (N, 3, 3), t in shape (N, 3, 1)
        :return: loss
        """
        R_pred, t_pred = pred
        R_target, t_target = target

        t_pred, t_target = t_pred.squeeze(dim=-1), t_target.squeeze(dim=-1)

        t_loss = torch.sqrt(torch.sum(torch.square(t_pred - t_target), dim=-1))
        t_loss = torch.mean(t_loss, dim=0)

        R_loss = torch.sqrt(torch.sum(torch.square(R_pred - R_target), dim=-1))
        R_loss = torch.mean(R_loss, dim=(0, 1))

        return self.alpha * R_loss + self.beta * t_loss
