import unittest
import matplotlib.pyplot as plt

from diffusion_ik import *
from affine_loss import *
from fk import *


class DiffusionModelTest(unittest.TestCase):

    def test_alpha(self):
        diffusion_model = DiffusionModel('cpu', 1000, 1e-4, 0.01)
        plt.plot(diffusion_model.alphas_cumprod)
        plt.show()


class RandomDiffusionIKDatasetTest(unittest.TestCase):

    def test_len(self):
        dataset = RandomDiffusionIKDataset('cpu', 10, 10)
        self.assertEqual(100, len(dataset))

    def test_beta(self):
        dataset = RandomDiffusionIKDataset('cpu', 10, 10)
        steps = torch.tensor([1, 2, 3, 4, 5])
        self.assertEqual(torch.Size([5]), dataset.beta(steps).shape)

    def test_getitem(self):
        dataset = RandomDiffusionIKDataset('cpu', 10, 10, beta_small=0, beta_large=0)
        x_t, noise = dataset.__getitem__(0)

        self.assertEqual(torch.Size([10, 13]), x_t.shape)
        self.assertEqual(torch.Size([10, 13]), x_t.shape)

        fk = FK('cpu')
        R_joints, t_joints = fk(x_t[:, :7])
        R_cartesian, t_cartesian = TransformationUtility.xyz_rpy_to_torch_affine(x_t[:, 7:])
        affine_loss = AffineLoss(alpha=1.0, beta=1.0)
        self.assertTrue(
            torch.allclose(affine_loss((R_joints, t_joints), (R_cartesian, t_cartesian)), torch.tensor(0.0), atol=1e-3))
