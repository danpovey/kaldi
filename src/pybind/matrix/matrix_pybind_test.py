#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import numpy as np

import kaldi


class TestFloatSubMatrix(unittest.TestCase):

    def test_from_numpy(self):
        num_rows = 5
        num_cols = 6
        data = np.arange(num_rows * num_cols).reshape(
            num_rows, num_cols).astype(np.float32)

        # =============================================================
        # build a FloatSubMatrix() from a numpy array; memory is shared
        # -------------------------------------------------------------
        m = kaldi.FloatSubMatrix(data)

        self.assertEqual(m.NumRows(), num_rows)
        self.assertEqual(m.NumCols(), num_cols)
        self.assertEqual(m.Stride(), data.strides[0] / 4)
        for r in range(num_rows):
            for c in range(num_cols):
                self.assertEqual(m[r, c], data[r, c])

        # memory is shared between numpy array and FloatSubMatrix
        for r in range(num_rows):
            for c in range(num_cols):
                m[r, c] += 10
                self.assertEqual(m[r, c], data[r, c])

        # =============================================================
        # Convert a FloatSubMatrix to a numpy array; memory is shared
        # -------------------------------------------------------------
        d = m.numpy()
        d += 10  # m is also changed because of memory sharing
        for r in range(num_rows):
            for c in range(num_cols):
                self.assertEqual(d[r, c], m[r, c])


class TestFloatMatrix(unittest.TestCase):

    def test_to_numpy(self):
        # first, build a kaldi matrix
        num_rows = 6
        num_cols = 8
        m = kaldi.FloatMatrix(row=num_rows, col=num_cols)
        for r in range(num_rows):
            for c in range(num_cols):
                self.assertEqual(m[r, c], 0)

        # now to numpy; memory is shared
        d = m.numpy()

        d += 10
        for r in range(num_rows):
            for c in range(num_cols):
                self.assertEqual(d[r, c], m[r, c])


if __name__ == '__main__':
    unittest.main()
