import torch
from typing import Tuple, Optional, Dict
from torch.utils.data import Dataset

from common import JointValuesScalerInverse
from fk import FK
from probabilistic_ik_config import ProbabilisticIKConfig


class ProbabilisticIKDataset(Dataset):
    def __init__(self, device, batch_size, batch_count, config: ProbabilisticIKConfig):
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.scaler_inverse = JointValuesScalerInverse(device)
        self.fk = FK(device)

    def __len__(self):
        return self.batch_count * self.batch_size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        :param index: index of the batch
        :return: dictionary with keys:
        joints
        pose(batch_size, 7, 12)
        joints_one_hot(batch_size, 7, config.trace_steps)
        """
        result = dict()

        joints_scaled = torch.rand(self.batch_size, 7, device=self.device)

        joints_trace_index = joints_scaled * self.config.trace_steps
        joints_trace_index = joints_trace_index.long()
        joints_one_hot = torch.zeros(self.batch_size, 7, self.config.trace_steps, device=self.device)
        joints_one_hot.scatter_(2, joints_trace_index.unsqueeze(-1), 1)
        result["joints_one_hot"] = joints_one_hot

        joints = self.scaler_inverse(joints_scaled)
        result["joints"] = joints

        pose_R, pose_t = [], []
        for i in range(7):
            R, t = self.fk(joints, begin=i)
            pose_R.append(R.view(-1, 9))
            pose_t.append(t.view(-1, 3))
        pose_R, pose_t = torch.stack(pose_R, dim=1), torch.stack(pose_t, dim=1)
        pose = torch.cat([pose_R, pose_t], dim=-1)

        result["pose"] = pose

        return result


# config = ProbabilisticIKConfig()
# datasets = ProbabilisticIKDataset(torch.device('cuda'), 2048, 1, config)
# print(datasets[0]["joints_one_hot"])
