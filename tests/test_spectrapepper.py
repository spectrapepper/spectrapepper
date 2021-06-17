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
        self.assertEqual(spep.load_targets()[0][0], 1.0155)
        self.assertEqual(spep.load_params()[0][0], 300)
        self.assertEqual(spep.load_mapp1()[0][0], 54.5611)
        self.assertEqual(spep.load_mapp2()[0][0], 54.8098)
      
    def test_loaders(self):
        
        data2 = spep.loadheader('datasets/headers.txt',2, split=True)
        r = data2[2]
        print('loadheader: ' + str(r))
        self.assertEqual(r, 'second')
        
        data2 = spep.loadline('datasets/spectras.txt',2)
        r = round(sum(data2),2)
        print('loadline: ' + str(r))
        self.assertEqual(r, 1460.99)
        
    
    def test_functions(self):
        spectras = spep.load_spectras()
        data = spectras[1]
        axis = spectras[0]
        data_l1 = spectras[2]
        params = spep.load_params(True)
        
        
        a = [0,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
        b = [4,2,2,3,4,3,2,3,4,1,0,3,1,2,2,3,0]
        c = [0,2,4,1,4,1,3,2,4,4,0,4,2,4,3,4,4]
        t = [0.69, 0.72, 0.83, 0.1, 0.13, 0.18, 0.13, 0.83, 0.16, 0.25,
              0.44, 0.42, 0.84, 0.16, 0.2, 0.71, 0.06, 0.16, 0.17, 0.42]
        
        
        data2 = spep.lowpass(data, cutoff=0.7, fs=10)
        r = np.floor(sum(data2))
        print('lowpass: ' + str(r))
        self.assertEqual(r, 1464)
        
        data2 = spep.moveavg(data, 10)
        r = np.floor(sum(data2))
        print('moveavg: ' + str(r))
        self.assertEqual(r, 1442)
        
        data2 = spep.normtomax(data)
        r = np.floor(sum(data2))
        print('normtomax: ' + str(r))
        self.assertEqual(r, 105)
        
        data2 = spep.normtovalue(data,100)
        r = np.floor(sum(data2))
        print('normtovalue: ' + str(r))
        self.assertEqual(r, 14)
        
        data2 = spep.alsbaseline(data)
        r = np.floor(np.floor(sum(data2)))
        print('alsbaseline: ' + str(r))
        self.assertEqual(r, 252)
        
        data2 = spep.bspbaseline(data, axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(sum(data2))
        print('bspbaseline: ' + str(r))
        self.assertEqual(r, 149)
        
        data2 = spep.polybaseline(data, axis, points=[100, 350, 700, 800], plot=False)
        r = np.floor(sum(data2))
        print('polybaseline: ' + str(r))
        self.assertEqual(r, 47)
        
        data2 = spep.lorentzfit(data, axis, plot=False)
        r = np.floor(sum(data2))
        print('lorentzfit: ' + str(r))
        self.assertEqual(r, 187)
        
        data2 = spep.gaussfit(data, axis, plot=False)
        r = np.floor(sum(data2))
        print('gaussfit: ' + str(r))
        self.assertEqual(r, 166)
        
        data2 = spep.cortopos([50,100,121,400], axis)
        r = np.floor(sum(data2))
        print('cortopos: ' + str(r))
        self.assertEqual(r, 523)
        
        data2 = spep.areacalculator(data, [[50, 100], [350, 450], [460, 500]], norm=True)
        r = round((sum(data2)),1)
        print('areacalculator: ' + str(r))
        self.assertEqual(r, 0.2)
        
        data2 = spep.bincombs(3)
        r = data2[2]
        print('bincombs: ' + str(r))
        self.assertEqual(r, (1, 1, 0))
        
        data2 = spep.normsum(data)
        r = np.floor(sum(data2))
        print('normsum: ' + str(r))
        self.assertEqual(r, 1)
        
        data2 = spep.normtoglobalmax(data)
        r = np.floor(sum(data2))
        print('normtoglobalmax: ' + str(r))
        self.assertEqual(r, 1)
        
        data1 = spep.load_mapp1()
        axis1 = data1[0]
        data2 = spep.load_mapp2()
        axis2 = data2[0]
        interpol = spep.interpolation([data1[1:],data2[1:]],[axis1,axis2])
        r = np.floor(sum(interpol[0][0]))
        print('interpolation: ' + str(r))
        self.assertEqual(r, 424112)
        
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
        
        data2 = spep.normtopeak(data,axis, 200)
        r = np.floor(sum(data2))
        print('normtopeak: ' + str(r))
        self.assertEqual(r, 460)
        
        data2 = spep.peakfinder(data)
        r = np.floor(sum(data2))
        print('peakfinder: ' + str(r))
        self.assertEqual(r, 35)
        
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
        
        data2 = spep.peakfit(data,axis,200)
        r = np.floor(sum(data2))
        print('peakfit: ' + str(r))
        self.assertEqual(r, 2834)
        
        data2 = spep.decbound([1,2,3],[3,4,5],[0,5],[0,5])
        r = np.floor(sum(data2[3]))
        print('decbound: ' + str(r))
        self.assertEqual(r, 2)
        
        data2 = spep.regression(a,[b,c])
        r = np.floor(data2[0][0])
        print('regression: ' + str(r))
        self.assertEqual(r, 1)
        
        data2 = spep.decdensity([-2,6,-1,6],a,b,c,0.25)
        r = np.floor(np.sum(data2))
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
        
        data2 = spep.trim(a, start=1, finish=9)
        r = np.floor(sum(data2))
        print('trim: ' + str(r))
        self.assertEqual(r, 20)
        
        data2 = spep.shuffle([a,b,c], delratio=0)
        r = sum(data2[1])
        print('shuffle: ' + str(r))
        self.assertEqual(r, 39)
        
        data2 = spep.mergedata([a,b,c])
        r = np.floor(np.sum(data2))
        print('mergedata: ' + str(r))
        self.assertEqual(r, 125)
        
        data2 = spep.logo(plot=False)
        r = data2
        print('shiftref: ' + str(r))
        self.assertEqual(r, False)
        
        data2 = spep.shiftref([data,data],[axis,axis],plot=False)
        r = np.floor(data2)
        print('shiftref: ' + str(r))
        self.assertEqual(r, 314)
        
        data2 = spep.classify(t, gnumber=4)
        r = sum(data2[0])
        print('classify1: ' + str(r))
        self.assertEqual(r, 30)
        
        data2 = spep.classify(t, gnumber=0, glimits=[0.25,0.5,0.75])
        r = sum(data2[0])
        print('classify2: ' + str(r))
        self.assertEqual(r, 19)
        
        data2 = spep.subtractref(a,b,alpha=0.5, plot=False)
        r = np.floor(np.sum(data2))
        print('subtractref: ' + str(r))
        self.assertEqual(r, 20)
        
        data2 = spep.pearson(params, plot=False)
        r = np.floor(np.sum(data2))
        print('pearson: ' + str(r))
        self.assertEqual(r, 25)
        
        data2 = spep.spearman(params, plot=False)
        r = np.floor(np.sum(data2))
        print('spearman: ' + str(r))
        self.assertEqual(r, 17)
        
        data2 = spep.grau(params, plot=False)
        r = np.floor(np.sum(data2))
        print('grau: ' + str(r))
        self.assertEqual(r, 12376)
        
        data2 = spep.moveavg(a,2)
        r = np.floor(np.sum(data2))
        print('moveavg: ' + str(r))
        self.assertEqual(r, 32)
        
        df = pd.DataFrame(data = np.transpose([a,b,c]), columns = ['D1', 'D2', 'T'])
        data2 = spep.plot2dml(df, plot=False)
        print('plot2dml: ')
        self.assertEqual(data2, False)
        
        data2 = spep.stackplot([data, data_l1], add=1, plot=False)
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
        r = np.floor(sum(data2))
        print('makeaxisstep: ' + str(r))
        self.assertEqual(r, 1489510)
        
        data2 = spep.makeaxisdivs(1000, 1500, 900, rounded=2)
        r = np.floor(sum(data2))
        print('makeaxisdivs: ' + str(r))
        self.assertEqual(r, 569305)
        
        
if __name__ == '__main__':
    unittest.main()
