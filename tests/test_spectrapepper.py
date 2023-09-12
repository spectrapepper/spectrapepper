#!/usr/bin/env python

"""Tests for `spectrapepper` package."""

import unittest
# import functions as spep
import spectrapepper as spep
import numpy as np
import pandas as pd

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
        axis, spectras = spep.load_spectras()
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
        print('typical: ', r)
        self.assertEqual(r, 1627)

        data2 = spep.lowpass(spectras[0], cutoff=0.7, fs=10)
        r = np.floor(sum(data2))
        print('lowpass: ', r)
        self.assertEqual(r, 1464)

        data2 = spep.normtomax(spectras[0], zeromin=True)
        r = round(sum(data2), 2)
        print('normtomax_zeromin: ', r)
        self.assertEqual(r, 89.49)

        data2 = spep.normtomax([spectras[0], spectras[1]], zeromin=True)
        r = round(np.sum(data2), 2)
        print('normtomax_multi_zeromin: ', r)
        self.assertEqual(r, 175.20)

        data2 = spep.normtoratio(spectras[0], r1=[100, 120], r2=[0, 500], x=axis)
        r = round(np.sum(data2), 2)
        print('normtoratio: ', r)
        self.assertEqual(r, 5.47) # 20.49

        data2 = spep.normtovalue(spectras[0],100)
        r = np.floor(sum(data2))
        print('normtovalue: ', r)
        self.assertEqual(r, 14)

        data2 = spep.normtovalue([spectras[0], spectras[1]],100)
        r = np.floor(np.sum(data2))
        print('normtovalue_multi: ', r)
        self.assertEqual(r, 29)

        data2 = spep.alsbaseline(spectras[0])
        r = (np.floor(sum(data2)))
        print('alsbaseline: ', r)
        self.assertEqual(r, 252)

        data2 = spep.alsbaseline([spectras[0], spectras[1]])
        r = (np.floor(np.sum(data2)))
        print('alsbaseline_multi: ', r)
        self.assertEqual(r, 515)

        data2 = spep.bspbaseline(y=spectras[0], x=axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(sum(data2))
        print('bspbaseline: ', r)
        self.assertEqual(r, 150)

        data2 = spep.bspbaseline(y=[spectras[0], spectras[1]], x=axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(np.sum(data2))
        print('bspbaseline_multi: ', r)
        self.assertEqual(r, 284)

        data2 = spep.polybaseline(spectras[0], axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(sum(data2))
        print('polybaseline: ', r)
        self.assertEqual(r, 46)

        data2 = spep.polybaseline([spectras[0], spectras[1]], axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(np.sum(data2))
        print('polybaseline_multi: ', r)
        self.assertEqual(r, 87)

        data2 = spep.lorentzfit(y=spectras[0], x=axis, pos=205, look=10)
        r = np.floor(sum(data2))
        print('lorentzfit: ', r)
        self.assertEqual(r, 232)

        data2 = spep.gaussfit(spectras[0], x=axis, pos=205, look=10)
        r = np.floor(sum(data2))
        print('gaussfit: ', r)
        self.assertEqual(r, 170)

        data2 = spep.studentfit(spectras[0], x=axis, pos=205, look=10)
        r = np.floor(sum(data2))
        print('studentfit: ', r)
        self.assertEqual(r, 151)

        data2 = spep.voigtfit(spep.normtomax(spectras[0]), x=axis, pos=205, look=10)
        r = np.floor(sum(data2))
        print('voigtfit: ', r)
        self.assertEqual(r, 16)

        data2 = spep.valtoind([50,100,121,400], axis)
        r = sum(data2)
        print('cortopos: ', r)
        self.assertEqual(r, 523)

        data2 = spep.valtoind([[50,100,121,400],[60,120,141,300]], axis)
        r = np.sum(data2)
        print('cortopos_multi: ', r)
        self.assertEqual(r, 992)

        data2 = spep.areacalculator(spectras[0], axis, [[50, 100], [350, 450], [460, 500]], norm=True)
        r = round(sum(data2),2)
        print('areacalculator: ', r)
        self.assertEqual(r, 0.23)

        data2 = spep.areacalculator([spectras[0], spectras[1]], axis, [[50, 100], [350, 450], [460, 500]], norm=True)
        r = round(np.sum(data2),2)
        print('areacalculator_multi: ', r)
        self.assertEqual(r, 0.46)

        data2 = spep.bincombs(3)
        r = data2[2]
        print('bincombs: ', r)
        self.assertEqual(r, (1, 1, 0))

        data2 = spep.normsum(spectras[0])
        r = round(sum(data2), 2)
        print('normsum: ', r)
        self.assertEqual(r, 1.00)

        data2 = spep.normsum([spectras[0], spectras[1]])
        r = round(np.sum(data2), 2)
        print('normsum_multi: ', r)
        self.assertEqual(r, 2.00)

        data2 = spep.normtoglobalmax(spectras[0])
        r = round(sum(data2), 2)
        print('normtoglobalmax: ', r)
        self.assertEqual(r, 105.12)

        data2 = spep.normtoglobalmax([spectras[0], spectras[1]])
        r = round(np.sum(data2), 2)
        print('normtoglobalmax_multi: ', r)
        self.assertEqual(r, 200.22)

        inter1, inter2 = spep.interpolation(spectras[0], axis)
        r = np.floor(sum(inter1))
        print('interpolation: ', r)
        self.assertEqual(r, 1347)

        data2 = spep.evalgrau([params[0],params[1],params[2]])[0]
        r = np.floor(sum(data2))
        print('evalgrau: ', r)
        self.assertEqual(r, 3)

        data2 = spep.groupscores(a,b,c)
        r = np.floor(sum(data2))
        print('groupscores: ', r)
        self.assertEqual(r, 1)

        data2 = spep.cmscore(a,b,c)
        r = np.floor(sum(data2[2]))
        print('cmscores: ', r)
        self.assertEqual(r, 10)

        data2 = spep.mdscore(a,b,c)
        r = sum(data2[2])
        print('mdscores: ', r)
        self.assertEqual(r, 11)

        data2 = spep.normtopeak(spectras[0], axis, 205)
        r = round(sum(data2), 2)
        print('normtopeak: ', r)
        self.assertEqual(r, 105.12)

        data2 = spep.normtopeak([spectras[0], spectras[1]], axis, 205)
        r = round(np.sum(data2), 2)
        print('normtopeak_multi: ', r)
        self.assertEqual(r, 205.14)

        data2 = spep.peakfinder(spectras[0], x=axis, ranges=[180, 220], look=10)
        print('peakfinder: ', r)
        self.assertEqual(data2, 172)

        data2 = spep.confusionmatrix(a,b)
        r = np.floor(np.sum(data2))
        print('confusionmatrix: ', r)
        self.assertEqual(r, 5)
        
        data2 = spep.avg([spectras[0],spectras[1]])
        r = np.floor(sum(data2))
        print('avg: ', r)
        self.assertEqual(r, 1462)

        data2 = spep.sdev([spectras[0],spectras[1]])
        r = np.floor(sum(data2))
        print('sdev: ', r)
        self.assertEqual(r, 21)

        data2 = spep.median([spectras[0],spectras[1]])
        r = np.floor(sum(data2))
        print('median: ', r)
        self.assertEqual(r, 1462)

        data2 = spep.decbound([1,3,5], [1,3,5], [0,1,2], [0,6,0,6], divs=1)
        r = np.floor(np.sum(data2))
        print('decbound: ', r)
        self.assertEqual(r, 24)

        data2 = spep.regression(a,[b,c])
        r = round(np.sum(data2[0]), 2)
        print('regression: ', r)
        self.assertEqual(r, 40)

        data2 = spep.regression(a, [b,c], cov=1)
        r = round(np.sum(data2[0]), 2)
        print('regression_cov: ', r)
        self.assertEqual(r, 136)

        data2 = spep.decdensity(a, b, c, divs=0.5, th=0.5)
        r = np.sum(data2)
        print('decdensity: ', r)
        self.assertEqual(r, 12)

        data2 = spep.isaxis([a,b,c])
        r = data2
        print('isaxis: ', r)
        self.assertEqual(r, False)

        data2 = spep.trim(spectras[0], start=1, finish=9)
        r = round(sum(data2), 2)
        print('trim: ', r)
        self.assertEqual(r, 1450.55)

        data2 = spep.trim([spectras[0],spectras[1]], start=1, finish=9)
        r = round(np.sum(data2), 2)
        print('trim_multi: ', r)
        self.assertEqual(r, 2898.12)

        data2 = spep.shuffle([a,b,c], delratio=0)
        r = sum(data2[1])
        print('shuffle: ', r)
        self.assertEqual(r, 39)

        data2 = spep.shuffle([a,b,c], delratio=0.1)
        r = len(data2[1])
        print('shuffle_delratio: ', r)
        self.assertEqual(r, 16)

        data2 = spep.mergedata([a,b,c])
        r = np.floor(np.sum(data2))
        print('mergedata: ', r)
        self.assertEqual(r, 125)

        data2 = spep.shiftref(ref_data=spectras[0], ref_axis=axis, ref_peak=205, plot=False)
        r = np.round(data2, 2)
        print('shiftref: ', r)
        self.assertEqual(r, -0.67)

        data2 = spep.classify(t, gnumber=4)
        r = sum(data2[0])
        print('classify1: ', r)
        self.assertEqual(r, 30)

        data2 = spep.classify(t, gnumber=0, glimits=[0.25, 0.5, 0.75])
        r = sum(data2[0])
        print('classify2: ', r)
        self.assertEqual(r, 19)

        data2 = spep.subtractref(a,c,alpha=0.59, plot=False)
        r = round(sum(data2), 2)
        print('subtractref: ', r)
        self.assertEqual(r, 12.86)

        data2 = spep.subtractref([a,b],c,alpha=0.79, plot=False)
        r = round(np.sum(data2), 2)
        print('subtractref_multi: ', r)
        self.assertEqual(r, 6.32)

        data2 = spep.pearson(params, plot=False)
        r = round(np.sum(data2), 2)
        print('pearson: ', r)
        self.assertEqual(r, 25.39)

        data2 = spep.spearman(params, plot=False)
        r = round(np.sum(data2), 2)
        print('spearman: ', r)
        self.assertEqual(r, 17.64)

        data2 = spep.grau(params, plot=False)
        r = np.floor(np.sum(data2))
        print('grau: ', r)
        self.assertEqual(r, 12376)

        data2 = spep.moveavg(spectras[0], 2)
        r = round(np.sum(data2), 2)
        print('moveavg: ', r)
        self.assertEqual(r, 1463.69)

        data2 = spep.moveavg([spectras[0],spectras[1]], 2)
        r = round(np.sum(data2), 2)
        print('moveavg_multi: ', r)
        self.assertEqual(r, 2924.63)

        df = pd.DataFrame(data = np.transpose([a,b,c]), columns = ['D1', 'D2', 'T'])
        data2 = spep.plot2dml(df, plot=False)
        print('plot2dml: ' + str(data2))
        self.assertEqual(data2, False)

        data2 = spep.stackplot([spectras[0], spectras[1]], offset=1, plot=False)
        print('stackplot: ')
        self.assertEqual(data2, False)

        data2 = spep.cosmicmp([spectras[0],spectras[0]])
        r = np.floor(np.sum(data2))
        print('cosmicmp: ', r)
        self.assertEqual(r, 2927)

        data2 = spep.cosmicdd([spectras[0],spectras[0]])
        r = np.floor(np.sum(data2))
        print('cosmicdd: ', r)
        self.assertEqual(r, 2927)

        data2 = spep.cosmicmed([spectras[0],spectras[0]])
        r = np.floor(np.sum(data2))
        print('cosmicmed: ', r)
        self.assertEqual(r, 2927)

        data2 = spep.makeaxisstep(1000, 0.98, 1000, rounded=2)
        r = round(sum(data2), 2)
        print('makeaxisstep: ', r)
        self.assertEqual(r, 1489510.00)

        data2 = spep.makeaxisstep(1000, 0.543, 1000, rounded=0)
        r = round(sum(data2), 2)
        print('makeaxisstep: ', r)
        self.assertEqual(r, 1271229.00)

        data2 = spep.makeaxisstep(1000, 0.789, 1000, adjust=True)
        r = round(sum(data2), 2)
        print('makeaxisstep: ', r)
        self.assertEqual(r, 1394105.50)

        data2 = spep.makeaxisdivs(1000, 1500, 900, rounded=2)
        r = round(sum(data2), 2)
        print('makeaxisdivs: ', r)
        self.assertEqual(r, 1125000.00)

        data2 = spep.makeaxisdivs(1000, 1400, 900, rounded=0)
        r = round(sum(data2), 2)
        print('makeaxisdivs: ', r)
        self.assertEqual(r, 1080000.00)

        data2 = spep.minmax(spectras)
        r = round(np.sum(data2), 2)
        print('minmax: ', r)
        self.assertEqual(r, 8399.54)

        data2 = spep.fwhm(y=spectras, x=axis, peaks=206, s=10)
        r = round(sum(data2), 2)
        print('fwhm: ', r)
        self.assertEqual(r, 2520.99)

        data2 = spep.fwhm(y=spectras[0], x=axis, peaks=206, s=10)
        r = round(np.sum(data2), 2)
        print('fwhm_single: ', r)
        self.assertEqual(r, 9.22)

        data2 = spep.asymmetry(y=spectras[0], x=axis, peak=206, s=5, limit=10)
        r = round(np.sum(data2), 2)
        print('asymmetry: ', r)
        self.assertEqual(r, -0.38)

        data3 = np.vstack((spectras[2],spectras[1],spectras[0]))
        data2 = spep.rwm(data3, (3,3))
        r = round(np.sum(data2), 2)
        print('running_median_nd: ', r)
        self.assertEqual(r, 1447.84)

        data2 = spep.typical(spectras[1:10])
        r = int(sum(data2))
        print('typical: ', r)
        self.assertEqual(r, 1626)

        data2 = spep.issinglevalue(spectras[0:3])
        print('issinglevalue: ' + str(data2))
        self.assertEqual(data2, [False, False, False])

        data2 = spep.mahalanobis([a[:2], b[:2], c[:2], d[:2]])
        r = round(sum(data2), 2)
        print('mahalanobis: ', r)
        self.assertEqual(r, 5.36)

        data2 = spep.representative(spectras[0:10])
        r = round(sum(data2), 2)
        print('representative: ', r)
        self.assertEqual(r, 1485.47)

        data2 = spep.autocorrelation(spectras[0], x=axis, lag=1)
        r = round(data2, 2)
        print('autocorrelation: ', r)
        self.assertEqual(r, 0.99)

        data2 = spep.crosscorrelation(y1=spectras[0], y2=spectras[1], lag=1)
        r = round(data2, 2)
        print('crosscorrelation: ', r)
        self.assertEqual(r, 4296.45)

        data2 = spep.derivative(spectras[0], x=None, s=1, deg=1)
        r = round(sum(data2), 2)
        print('derivative: ', r)
        self.assertEqual(r, -1.5)

        data2 = spep.peaksimilarity(y1=spectras[0], y2=spectras[1], p1=[206], p2=[206], x=axis, plot=False)
        r = round(data2[0][0], 2)
        print('peaksimilarity: ', r)
        self.assertEqual(r, 0.99)

        data2 = spep.reverse(y=spectras[0])
        r = round(sum(data2), 2)
        print('reverse: ', r)
        self.assertEqual(r, 1463.76)

        data2 = spep.count(y=a, value=[0,1])
        print('count: ', data2)
        self.assertEqual(data2, [1, 4])

        
        data2 = spep.vectortoimg(y=a)
        print('vectortoimg: ', np.array(data2).shape)
        self.assertEqual(np.array(data2).shape, (17, 17))
        
        data2 = spep.deconvolution(y=spectras[0], x=axis, pos=175)
        r = int(sum(data2))
        print('deconvolution: ', r)
        self.assertEqual(r, 1201)
        


if __name__ == '__main__':
    unittest.main()
