#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

try:
    import torch
except ImportError:
    print('This test need PyTorch.')
    print('please install PyTorch first.')
    print('PyTorch 1.3.0dev20191006 has been tested.')
    sys.exit(0)

from torch.utils.dlpack import to_dlpack

import kaldi_pybind.fst as fst

import kaldi_pybind
import kaldi


class TestDLPack(unittest.TestCase):

    def test_pytorch_cpu_tensor_to_subvector(self):

        tensor = torch.arange(3).float()

        # v and tensor share the same memory
        # no data is copied
        v = kaldi.to_subvector(to_dlpack(tensor))
        self.assertIsInstance(v, kaldi_pybind.FloatSubVector)

        v[0] = 100
        v[1] = 200
        v[2] = 300

        self.assertEqual(v[0], 100)
        self.assertEqual(v[1], 200)
        self.assertEqual(v[2], 300)

    def test_pytorch_cpu_tensor_to_submatrix(self):
        tensor = torch.arange(6).reshape(2, 3).float()

        m = kaldi.to_submatrix(to_dlpack(tensor))
        self.assertIsInstance(m, kaldi_pybind.FloatSubMatrix)

        m[0, 0] = 100  # also changes tensor, since memory is shared
        self.assertEqual(tensor[0, 0], 100)


if __name__ == '__main__':
    unittest.main()
