#!/usr/bin/env python

"""Tests for `spectrapepper` package."""

import unittest
import spectrapepper as spep


class TestSpectrapepper(unittest.TestCase):
    """Tests for `spectrapepper` package."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_datasets(self):
        self.assertEqual(spep.load_spectras()[0][0], 47.0712)
        self.assertEqual(spep.load_targets()[0][0], 1.0155)
        self.assertEqual(spep.load_params()[0][0], 300)

if __name__ == '__main__':
    unittest.main()
