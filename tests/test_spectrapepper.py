#!/usr/bin/env python

"""Tests for `spectrapepper` package."""

import unittest
import spectrapepper as spep
import numpy as np
import pandas as pd
# import my_functions as spep
# import sys

class TestSpectrapepper(unittest.TestCase):
    """Tests for `spectrapepper` package."""
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_datasets(self):
        self.assertEqual(spep.load_spectras()[0][0], 47.0712)
        self.assertEqual(spep.load_targets()[0], 1.0155)
        self.assertEqual(spep.load_params()[0][0], 300)

    def test_functions(self):
        spectras = spep.load_spectras()
        data = spectras[1]
        axis = spectras[0]
        data_l1 = spectras[2]
        params = spep.load_params(True)

        a = [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        b = [4, 2, 2, 3, 4, 3, 2, 3, 4, 1, 0, 3, 1, 2, 2, 3, 0]
        c = [0, 2, 4, 1, 4, 1, 3, 2, 4, 4, 0, 4, 2, 4, 3, 4, 4]
        d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        t = [0.69, 0.72, 0.83, 0.1, 0.13, 0.18, 0.13, 0.83, 0.16, 0.25,
              0.44, 0.42, 0.84, 0.16, 0.2, 0.71, 0.06, 0.16, 0.17, 0.42]

        data2 = spep.issinglevalue(y=[a, b, d])
        self.assertEqual(data2, [False, False, True])

        data2 = spep.typical(spectras[:10])
        r = round(sum(data2))
        print('typical: ' + str(r))
        self.assertEqual(r, 1879)

        data2 = spep.lowpass(data, cutoff=0.7, fs=10)
        r = np.floor(sum(data2))
        print('lowpass: ' + str(r))
        self.assertEqual(r, 1464)

        data2 = spep.moveavg(data, 10)
        r = np.floor(sum(data2))
        print('moveavg: ' + str(r))
        self.assertEqual(r, 1442)

        data2 = spep.normtomax(data, zeromin=True)
        r = round(sum(data2), 2)
        print('normtomax_zeromin: ' + str(r))
        self.assertEqual(r, 89.49)

        data2 = spep.normtomax([data, data_l1], zeromin=True)
        r = round(np.sum(data2), 2)
        print('normtomax_multi_zeromin: ' + str(r))
        self.assertEqual(r, 175.20)

        data2 = spep.normtovalue(data,100)
        r = np.floor(sum(data2))
        print('normtovalue: ' + str(r))
        self.assertEqual(r, 14)

        data2 = spep.normtovalue([data, data_l1],100)
        r = np.floor(np.sum(data2))
        print('normtovalue_multi: ' + str(r))
        self.assertEqual(r, 29)

        data2 = spep.alsbaseline(data)
        r = (np.floor(sum(data2)))
        print('alsbaseline: ' + str(r))
        self.assertEqual(r, 252)

        data2 = spep.alsbaseline([data, data_l1])
        r = (np.floor(np.sum(data2)))
        print('alsbaseline_multi: ' + str(r))
        self.assertEqual(r, 515)

        data2 = spep.bspbaseline(y=data, x=axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(sum(data2))
        print('bspbaseline: ' + str(r))
        self.assertEqual(r, 149)

        data2 = spep.bspbaseline(y=[data, data_l1], x=axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(np.sum(data2))
        print('bspbaseline_multi: ' + str(r))
        self.assertEqual(r, 283)

        data2 = spep.polybaseline(data, axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(sum(data2))
        print('polybaseline: ' + str(r))
        self.assertEqual(r, 47)

        data2 = spep.polybaseline([data, data_l1], axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(np.sum(data2))
        print('polybaseline_multi: ' + str(r))
        self.assertEqual(r, 88)

        data2 = spep.lorentzfit(y=data, x=axis, pos=205)
        r = np.floor(sum(data2))
        print('lorentzfit: ' + str(r))
        self.assertEqual(r, 232)

        data2 = spep.gaussfit(data, x=axis, pos=205)
        r = np.floor(sum(data2))
        print('gaussfit: ' + str(r))
        self.assertEqual(r, 169)

        data2 = spep.valtoind([50,100,121,400], axis)
        r = sum(data2)
        print('cortopos: ' + str(r))
        self.assertEqual(r, 523)

        data2 = spep.valtoind([[50,100,121,400],[60,120,141,300]], axis)
        r = np.sum(data2)
        print('cortopos_multi: ' + str(r))
        self.assertEqual(r, 992)

        data2 = spep.areacalculator(data, axis, [[50, 100], [350, 450], [460, 500]], norm=True)
        r = round(sum(data2),2)
        print('areacalculator: ' + str(r))
        self.assertEqual(r, 0.23)

        data2 = spep.areacalculator([data, data_l1], axis, [[50, 100], [350, 450], [460, 500]], norm=True)
        r = round(np.sum(data2),2)
        print('areacalculator_multi: ' + str(r))
        self.assertEqual(r, 0.46)

        data2 = spep.bincombs(3)
        r = data2[2]
        print('bincombs: ' + str(r))
        self.assertEqual(r, (1, 1, 0))

        data2 = spep.normsum(data)
        r = round(sum(data2), 2)
        print('normsum: ' + str(r))
        self.assertEqual(r, 1.00)

        data2 = spep.normsum([data, data_l1])
        r = round(np.sum(data2), 2)
        print('normsum_multi: ' + str(r))
        self.assertEqual(r, 2.00)

        data2 = spep.normtoglobalmax(data)
        r = round(sum(data2), 2)
        print('normtoglobalmax: ' + str(r))
        self.assertEqual(r, 105.12)

        data2 = spep.normtoglobalmax([data, data_l1])
        r = round(np.sum(data2), 2)
        print('normtoglobalmax_multi: ' + str(r))
        self.assertEqual(r, 200.22)

        inter1, inter2 = spep.interpolation(data, axis)
        r = np.floor(sum(inter1))
        print('interpolation: ' + str(r))
        self.assertEqual(r, 1347)

        data2 = spep.evalgrau([params[0],params[1],params[2]])[0]
        r = np.floor(sum(data2))
        print('evalgrau: ' + str(r))
        self.assertEqual(r, 3)

        data2 = spep.groupscores(a,b,c)
        r = np.floor(sum(data2))
        print('groupscores: ' + str(r))
        self.assertEqual(r, 1)

        data2 = spep.cmscore(a,b,c)
        r = np.floor(sum(data2[2]))
        print('cmscores: ' + str(r))
        self.assertEqual(r, 10)

        data2 = spep.mdscore(a,b,c)
        r = sum(data2[2])
        print('mdscores: ' + str(r))
        self.assertEqual(r, 11)

        data2 = spep.normtopeak(data, axis, 205)
        r = round(sum(data2), 2)
        print('normtopeak: ' + str(r))
        self.assertEqual(r, 105.12)

        data2 = spep.normtopeak([data, data_l1], axis, 205)
        r = round(np.sum(data2), 2)
        print('normtopeak_multi: ' + str(r))
        self.assertEqual(r, 205.14)

        data2 = spep.peakfinder(data)
        r = sum(data2)
        print('peakfinder: ' + str(r))
        self.assertEqual(r, 19809)

        data2 = spep.confusionmatrix(a,b)
        r = np.floor(np.sum(data2))
        print('confusionmatrix: ' + str(r))
        self.assertEqual(r, 1)

        data2 = spep.avg([data,data_l1])
        r = np.floor(sum(data2))
        print('avg: ' + str(r))
        self.assertEqual(r, 1462)

        data2 = spep.sdev([data,data_l1])
        r = np.floor(sum(data2))
        print('sdev: ' + str(r))
        self.assertEqual(r, 30)

        data2 = spep.median([data,data_l1])
        r = np.floor(sum(data2))
        print('median: ' + str(r))
        self.assertEqual(r, 1462)

        data2 = spep.decbound([1,3,5], [1,3,5], [0,1,2], [0,6,0,6], divs=1)
        r = np.floor(np.sum(data2))
        print('decbound: ' + str(r))
        self.assertEqual(r, 24)

        data2 = spep.regression(a,[b,c])
        r = round(np.sum(data2[0]), 2)
        print('regression: ' + str(r))
        self.assertEqual(r, 40)

        data2 = spep.regression(a, [b,c], cov=1)
        r = round(np.sum(data2[0]), 2)
        print('regression_cov: ' + str(r))
        self.assertEqual(r, 136)

        data2 = spep.decdensity(a, b, c, divs=0.5, th=0.5)
        r = np.sum(data2)
        print('decdensity: ' + str(r))
        self.assertEqual(r, 12)

        data2 = spep.colormap(['red','green','blue'],'test',n=101)
        r = np.floor(data2(99)[1])
        print('colormap: ' + str(r))
        self.assertEqual(r, 0)

        data2 = spep.isaxis([a,b,c])
        r = data2
        print('isaxis: ' + str(r))
        self.assertEqual(r, False)

        data2 = spep.trim(data, start=1, finish=9)
        r = round(sum(data2), 2)
        print('trim: ' + str(r))
        self.assertEqual(r, 1450.55)

        data2 = spep.trim([data,data_l1], start=1, finish=9)
        r = round(np.sum(data2), 2)
        print('trim_multi: ' + str(r))
        self.assertEqual(r, 2898.12)

        data2 = spep.shuffle([a,b,c], delratio=0)
        r = sum(data2[1])
        print('shuffle: ' + str(r))
        self.assertEqual(r, 39)

        data2 = spep.shuffle([a,b,c], delratio=0.1)
        r = len(data2[1])
        print('shuffle_delratio: ' + str(r))
        self.assertEqual(r, 16)

        data2 = spep.mergedata([a,b,c])
        r = np.floor(np.sum(data2))
        print('mergedata: ' + str(r))
        self.assertEqual(r, 125)

        data2 = spep.shiftref(ref_data=data, ref_axis=axis, ref_peak=205, plot=False)
        r = np.round(data2, 2)
        print('shiftref: ' + str(r))
        self.assertEqual(r, -0.67)

        data2 = spep.classify(t, gnumber=4)
        r = sum(data2[0])
        print('classify1: ' + str(r))
        self.assertEqual(r, 30)

        data2 = spep.classify(t, gnumber=0, glimits=[0.25,0.5,0.75])
        r = sum(data2[0])
        print('classify2: ' + str(r))
        self.assertEqual(r, 19)

        data2 = spep.subtractref(a,c,alpha=0.59, plot=False)
        r = round(sum(data2), 2)
        print('subtractref: ' + str(r))
        self.assertEqual(r, 12.86)

        data2 = spep.subtractref([a,b],c,alpha=0.79, plot=False)
        r = round(np.sum(data2), 2)
        print('subtractref_multi: ' + str(r))
        self.assertEqual(r, 6.32)

        data2 = spep.pearson(params, plot=False)
        r = round(np.sum(data2), 2)
        print('pearson: ' + str(r))
        self.assertEqual(r, 25.39)

        data2 = spep.spearman(params, plot=False)
        r = round(np.sum(data2), 2)
        print('spearman: ' + str(r))
        self.assertEqual(r, 17.64)

        data2 = spep.grau(params, plot=False)
        r = np.floor(np.sum(data2))
        print('grau: ' + str(r))
        self.assertEqual(r, 12376)

        data2 = spep.moveavg(data, 2)
        r = round(np.sum(data2), 2)
        print('moveavg: ' + str(r))
        self.assertEqual(r, 1460.06)

        data2 = spep.moveavg([data,data_l1], 2)
        r = round(np.sum(data2), 2)
        print('moveavg_multi: ' + str(r))
        self.assertEqual(r, 2917.29)

        df = pd.DataFrame(data = np.transpose([a,b,c]), columns = ['D1', 'D2', 'T'])
        data2 = spep.plot2dml(df, plot=False)
        print('plot2dml: ' + str(data2))
        self.assertEqual(data2, False)

        data2 = spep.stackplot([data, data_l1], offset=1, plot=False)
        print('stackplot: ')
        self.assertEqual(data2, False)

        data2 = spep.cosmicmp([data,data])
        r = np.floor(np.sum(data2))
        print('cosmicmp: ' + str(r))
        self.assertEqual(r, 2927)

        data2 = spep.cosmicdd([data,data])
        r = np.floor(np.sum(data2))
        print('cosmicdd: ' + str(r))
        self.assertEqual(r, 2927)

        data2 = spep.cosmicmed([data,data])
        r = np.floor(np.sum(data2))
        print('cosmicmed: ' + str(r))
        self.assertEqual(r, 2927)

        data2 = spep.makeaxisstep(1000, 0.98, 1000, rounded=2)
        r = round(sum(data2), 2)
        print('makeaxisstep: ' + str(r))
        self.assertEqual(r, 1489510.00)

        data2 = spep.makeaxisstep(1000, 0.543, 1000, rounded=0)
        r = round(sum(data2), 2)
        print('makeaxisstep: ' + str(r))
        self.assertEqual(r, 1271229.00)

        data2 = spep.makeaxisstep(1000, 0.789, 1000, adjust=True)
        r = round(sum(data2), 2)
        print('makeaxisstep: ' + str(r))
        self.assertEqual(r, 1394105.50)

        data2 = spep.makeaxisdivs(1000, 1500, 900, rounded=2)
        r = round(sum(data2), 2)
        print('makeaxisdivs: ' + str(r))
        self.assertEqual(r, 1125000.00)

        data2 = spep.makeaxisdivs(1000, 1400, 900, rounded=0)
        r = round(sum(data2), 2)
        print('makeaxisdivs: ' + str(r))
        self.assertEqual(r, 1080000.00)

        data2 = spep.minmax(spectras)
        r = round(np.sum(data2), 2)
        print('minmax: ' + str(r))
        self.assertEqual(r, 532470.08)

        data2 = spep.fwhm(y=spectras, x=axis, peaks=250, s=10)
        r = round(sum(data2), 2)
        print('fwhm: ' + str(r))
        self.assertEqual(r, 2484.96)

        data2 = spep.fwhm(y=data, x=axis, peaks=250, s=10)
        r = round(np.sum(data2), 2)
        print('fwhm_single: ' + str(r))
        self.assertEqual(r, 1.84)

        data2 = spep.asymmetry(y=data, x=axis, peak=250, s=5, limit=10)
        r = round(np.sum(data2), 2)
        print('asymmetry: ' + str(r))
        self.assertEqual(r, -0.13)

        data3 = np.vstack((spectras[2],spectras[1],spectras[0]))
        data2 = spep.rwm(data3, (3,3))
        r = round(np.sum(data2), 2)
        print('running_median_nd: ' + str(r))
        self.assertEqual(r,1510.57)



if __name__ == '__main__':
    unittest.main()
