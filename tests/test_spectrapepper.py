#!/usr/bin/env python

"""Tests for `spectrapepper` package."""

import unittest
import functions as spep


class TestSpectrapepper(unittest.TestCase):
    """Tests for `spectrapepper` package."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_datasets(self):
        self.assertEqual(int(spep.load_spectras()[0][0]), 47)
        self.assertEqual(int(spep.load_targets()[0][0]), 47)
        self.assertEqual(int(spep.load_params()[0][0]), 47)

if __name__ == '__main__':
    unittest.main()
