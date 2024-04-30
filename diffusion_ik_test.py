import unittest

from diffusion_ik import *
from affine_loss import *
from fk import *


class RandomDiffusionIKDatasetTest(unittest.TestCase):

    def test_len(self):
        dataset = RandomDiffusionIKDataset('cpu', 10, 10)
        self.assertEqual(100, len(dataset))

    def test_getitem(self):
        dataset = RandomDiffusionIKDataset('cpu', 10, 10)
        x, y = dataset.__getitem__(0)

        self.assertEqual(torch.Size([10, 13]), x.shape)
        self.assertEqual(torch.Size([10, 13]), y.shape)

        fk = FK('cpu')
        R_joints, t_joints = fk(y[:, :7])
        R_cartesian, t_cartesian = TransformationUtility.xyz_rpy_to_torch_affine(y[:, 7:])
        affine_loss = AffineLoss(alpha=1.0, beta=1.0)
        self.assertTrue(
            torch.allclose(affine_loss((R_joints, t_joints), (R_cartesian, t_cartesian)), torch.tensor(0.0), atol=1e-3))
