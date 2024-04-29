import unittest

import torch

from fk import *


class TransformationUtilityTest(unittest.TestCase):

    def test_xyz_rpy_to_torch_affine(self):
        pass


class FKTest(unittest.TestCase):
    def test_zero_state(self):
        fk = FK('cpu')
        zero_state = torch.zeros(1, 7)
        R_result, t_result = fk(zero_state)

        self.assertEqual(torch.Size([1, 3, 3]), R_result.shape)
        self.assertEqual(torch.Size([1, 3, 1]), t_result.shape)

        target_position = torch.Tensor([[[2.0600e-01], [3.3539e-06], [1.2050e-01]]])

        self.assertTrue(torch.allclose(target_position, t_result, atol=1e-3))

    def test_initial_state(self):
        fk = FK('cpu')
        initial_state = torch.Tensor([[0, 0.262, -3.142, 2.618, 0, 1.309, 0]])
        R_result, t_result = fk(initial_state)

        self.assertEqual(torch.Size([1, 3, 3]), R_result.shape)
        self.assertEqual(torch.Size([1, 3, 1]), t_result.shape)

        target_position = torch.Tensor([[[-0.086], [0], [0.990]]])

        self.assertTrue(torch.allclose(target_position, t_result, atol=1e-3))

    def test_forward_state(self):
        fk = FK('cpu')
        forward_state = torch.Tensor([[-3.142, 0.262, -3.046, 2.316, -0.192, 2.499, 0.710]])
        R_result, t_result = fk(forward_state)

        self.assertEqual(torch.Size([1, 3, 3]), R_result.shape)
        self.assertEqual(torch.Size([1, 3, 1]), t_result.shape)

        target_position = torch.Tensor([[[0.1746], [0.0067], [0.8052]]])

        self.assertTrue(torch.allclose(target_position, t_result, atol=1e-3))

    def test_multiple_states(self):
        fk = FK('cpu')
        states = torch.Tensor([
            [0, 0.262, -3.142, 2.618, 0, 1.309, 0],
            [-3.142, 0.262, -3.046, 2.316, -0.192, 2.499, 0.710]
        ])
        R_result, t_result = fk(states)

        self.assertEqual(torch.Size([2, 3, 3]), R_result.shape)
        self.assertEqual(torch.Size([2, 3, 1]), t_result.shape)

        target_positions = torch.Tensor([
            [[-0.086], [0], [0.990]],
            [[0.1746], [0.0067], [0.8052]]
        ])

        self.assertTrue(torch.allclose(target_positions, t_result, atol=1e-3))


class RandomFKDatasetTest(unittest.TestCase):
    def test_len(self):
        dataset = RandomFKDataset('cpu', 10, 10)
        self.assertEqual(100, len(dataset))

    def test_get_item(self):
        dataset = RandomFKDataset('cpu', 10, 10)
        joints, Rs, ts = dataset[0]
        self.assertEqual(torch.Size([10, 7]), joints.shape)
        self.assertEqual(torch.Size([10, 3, 3]), Rs.shape)
        self.assertEqual(torch.Size([10, 3, 1]), ts.shape)


if __name__ == '__main__':
    unittest.main()
