#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

from kaldi import SequentialNnetChainExampleReader
from kaldi import RandomAccessNnetChainExampleReader
from kaldi import NnetChainExampleWriter
import kaldi_pybind.nnet3 as nnet3


class TestNnetChainExample(unittest.TestCase):

    def test_nnet_chain_example(self):

        # TODO(fangjun): find a place to store the test data
        egs_rspecifier = 'scp:./aishell_test.scp'
        reader = SequentialNnetChainExampleReader(egs_rspecifier)
        for key, value in reader:
            # TODO(fangjun: fix "RuntimeError: Could not allocate list object"
            # print(value.inputs)
            self.assertTrue(isinstance(key, str))
            self.assertTrue(isinstance(value, nnet3.NnetChainExample))
            outputs = value.outputs
            num_outputs = len(outputs)
            self.assertEqual(num_outputs, 1)

            sup = outputs[0]
            self.assertTrue(isinstance(sup, nnet3.NnetChainSupervision))
            # TODO(fangjun): finish the test


if __name__ == '__main__':
    unittest.main()
