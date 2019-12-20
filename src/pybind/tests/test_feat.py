#!/usr/bin/env python3

# Copyright 2019   Microsoft Corporation (author: Xingyu Na)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest
import numpy as np

import kaldi_pybind as k

import kaldi_pybind.feat as feat
from kaldi import SequentialWaveReader
from kaldi import SequentialMatrixReader

class TestFeat(unittest.TestCase):

    def test_mfcc(self):
        mfcc = feat.Mfcc(feat.MfccOptions())
        reader = SequentialWaveReader('ark:tests/wav.ark')
        # gold set is feature extracted using featbin/compute-mfcc-feats
        gold_reader = SequentialMatrixReader('ark:tests/feat.ark')
        for key, value in reader:
            print("Validate utterance: {}".format(key))
            self.assertEqual(value.SampFreq(), 16000)

            wave_data = value.Data()
            nd = np.array(wave_data, copy=False)
            nsamp = wave_data.NumCols()
            self.assertAlmostEqual(nsamp, value.Duration() * value.SampFreq(), places=1)

            waveform = k.FloatSubVector(nd.reshape(nsamp))
            features = k.FloatMatrix(1, 1)
            mfcc.ComputeFeatures(waveform, value.SampFreq(), 1.0, features)
            self.assertEqual(key, gold_reader.Key())
            gold_feat = np.array(gold_reader.Value(), copy=False)
            self.assertTrue((np.array(features, copy=False) == gold_feat).all())
            gold_reader.Next()

if __name__ == '__main__':
    unittest.main()
