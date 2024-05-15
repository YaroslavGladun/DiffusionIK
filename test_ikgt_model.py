import unittest
import torch
import numpy as np
from seed_epsilon_ik_model import IndexMapper, OneHotEncoder, IKGTModel


class TestIndexMapper(unittest.TestCase):
    def setUp(self):
        self.index_mapper = IndexMapper()

    def test_forward(self):
        x = torch.tensor([0.0, np.pi / 2, np.pi, -np.pi / 2, -np.pi])
        expected_output = torch.tensor([499, 749, 999, 249, 0])

        output = self.index_mapper.forward(x)
        print(output)
        self.assertTrue(torch.equal(output, expected_output))


class TestOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.one_hot_encoder = OneHotEncoder()

    def test_forward(self):
        x = torch.tensor([0, 1, 2, 3, 4])
        expected_output = torch.zeros((5, 1000))
        expected_output[range(5), x] = 1

        output = self.one_hot_encoder.forward(x)
        self.assertTrue(torch.equal(output, expected_output))


class TestIKGTModel(unittest.TestCase):
    def setUp(self):
        self.ikgt_model = IKGTModel(device='cpu')

    def test_make_pair_for_model(self):
        x = torch.randn((10, 12))
        y = torch.randn((10, 7))
        model_index = 3

        x_new, y_new = self.ikgt_model.make_pair_for_model(x, y, model_index)

        self.assertEqual(x_new.shape, (10, 12 + model_index))
        self.assertEqual(y_new.shape, (10, 1000))

    def test_make_pair_for_model_where_index_0(self):
        x = torch.randn((10, 12))
        y = torch.randn((10, 7))
        model_index = 0

        x_new, y_new = self.ikgt_model.make_pair_for_model(x, y, model_index)

        self.assertEqual(x_new.shape, (10, 12))
        self.assertEqual(y_new.shape, (10, 1000))


if __name__ == '__main__':
    unittest.main()
