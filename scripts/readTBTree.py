
#!/usr/bin/env python

import ROOT
from pylab import *
import numpy as np
import pandas as pd
import multiprocessing as mp

class TBTreeReader():
    def __init__(self,fname, channels):
        self.fname = fname
        self.file = ROOT.TFile(fname)
        self.channels = channels
        
    def readTree(self):
        df = self.getEntries()
        return pd.DataFrame(df)
    
    def getEntries(self):
        tree = self.file.Get("TBTree")
        nev = int(tree.GetEntriesFast())
        for iev in range(nev):
            tree.GetEntry(iev)
            entry = {}
            entry["tposX_dut"] = self._stdvec_to_nparray(tree.tposX_dut)
            entry["tposY_dut"] = self._stdvec_to_nparray(tree.tposY_dut)          
            entry["dposX_dut"] = self._stdvec_to_nparray(tree.dposX_dut)
            entry["dposY_dut"] = self._stdvec_to_nparray(tree.dposY_dut)
            entry["dposX_quad"] = self._stdvec_to_nparray(tree.dposX_quad)
            entry["dposY_quad"] = self._stdvec_to_nparray(tree.dposY_quad)            
            entry["qposX"] = self._stdvec_to_nparray(tree.tposX_dut)
            entry["qposY"] = self._stdvec_to_nparray(tree.tposY_dut)
            
            entry["nTriples"] = entry["tposX_dut"].size
            entry["nDriples"] = entry["dposX_dut"].size
            entry["nQuad"]    = entry["qposX"].size

            nTDMatches = 0
            for i in range(entry["nTriples"]):
                for j in range(entry["nDriples"]):
                    dx = entry["dposX_dut"][j]-entry["tposX_dut"][i]
                    dy = entry["dposY_dut"][j]-entry["tposY_dut"][i]
                    dr = np.sqrt(dx*dx+dy*dy)
                    if dr < 0.1:
                        nTDMatches +=1
            entry["nTDMatches"] = nTDMatches
                      
            
            for backend in ["VME","uTCA"]:
                for ch in self.channels:
                    temp = self._getWaveFeature(tree, backend, ch)
                    entry["{}Ch{}_baselineMeam".format(backend,ch)]       = temp[0]
                    entry["{}Ch{}_baselineStd".format(backend, ch)]       = temp[1]
                    entry["{}Ch{}_pulsePos".format(backend, ch)]          = temp[2]
                    entry["{}Ch{}_pulseAmp".format(backend, ch)]          = temp[3]
                    entry["{}Ch{}_pulsePosPassCuts".format(backend, ch)]  = temp[4]
                    entry["{}Ch{}_pulseAmpPassCuts".format(backend, ch)]  = temp[5]
                    entry["{}Ch{}_pulseFWHM".format(backend, ch)]         = temp[6]
                    entry["{}Ch{}_pulseIntegral".format(backend, ch)]     = temp[7]
                    entry["{}Ch{}_pulseMAxSlope".format(backend, ch)]     = temp[8]

            
            yield entry

            if iev%10000 == 0:
                print("FILE {} -- {}k/{}k".format(self.fname,iev//1000,nev//1000) )
        
    
    def _getWaveFeature(self,tree, backend, ch):
        if backend=="uTCA":
            if ch == "0":
                wave = self._stdvec_to_nparray(tree.uTCA_CH0)
            if ch == "1":
                wave = self._stdvec_to_nparray(tree.uTCA_CH1)
            if ch == "2":
                wave = self._stdvec_to_nparray(tree.uTCA_CH2)
            if ch == "3":
                wave = self._stdvec_to_nparray(tree.uTCA_CH3)
            baselineMeam = wave[1000:].mean()   
            baselineStd = wave[1000:].std()
            pulsePosWindow0, pulsePosWindow1 = 270, 320
            
        
        if backend=="VME":
            if ch == "0":
                wave = self._stdvec_to_nparray(tree.VME_CH0)
            if ch == "1":
                wave = self._stdvec_to_nparray(tree.VME_CH1)
            if ch == "2":
                wave = self._stdvec_to_nparray(tree.VME_CH2)
            if ch == "3":
                wave = self._stdvec_to_nparray(tree.VME_CH3)
            baselineMeam = wave[:1000].mean()
            baselineStd = wave[:1000].std()
            pulsePosWindow0, pulsePosWindow1 = 1405, 1425
            


        wave = np.abs(wave - baselineMeam)
        pulsePos = np.argmax(wave)
        pulseAmp = np.max(wave)
        pulsePosPassCuts = (pulsePos>pulsePosWindow0) and (pulsePos<pulsePosWindow1)
        pulseAmpPassCuts = pulseAmp/baselineStd>5.1
        
        if pulsePosPassCuts and pulseAmpPassCuts:
            pulseFWHM = 0
            for ishift in range(0,100):
                iamp     = wave[pulsePos+ishift]
                iampNext = wave[pulsePos+ishift+1]
                if iamp>(pulseAmp/2) and iampNext<(pulseAmp/2):
                    shift = ishift + (iamp-pulseAmp/2)/(iamp-iampNext)
                    pulseFWHM += shift
                    break
            for ishift in range(0,100):
                iamp     = abs(wave[pulsePos-ishift] - baselineMeam )
                iampNext = abs(wave[pulsePos-ishift-1] - baselineMeam )
                if iamp>(pulseAmp/2) and iampNext<(pulseAmp/2):
                    shift = ishift + (iamp-pulseAmp/2)/(iamp-iampNext)
                    pulseFWHM += shift
                    break
            pulseWave     = wave[pulsePos-100:pulsePos+100]
            pulseIntegral = np.sum(pulseWave)
            pulseMaxSlope = np.max(np.abs(np.diff(pulseWave)))
        else:
            pulseFWHM, pulseIntegral, pulseMaxSlope = 0,0,0
        
        result = (baselineMeam,baselineStd,pulsePos,pulseAmp,
                  pulsePosPassCuts,pulseAmpPassCuts,
                  pulseFWHM,pulseIntegral,pulseMaxSlope)
        return result

    def _stdvec_to_nparray(self, stdvec):
        if stdvec.size() > 0:
            return np.asarray(stdvec,dtype=np.float32)
        else:
            return np.array([])
    
if __name__ == "__main__":
    def processFile(config):
      run, channels = config
      rd = TBTreeReader("/Users/zihengchen/Documents/BRIL/data/TBTree/TBTree{}.root".format(run),channels)
      rd.readTree().to_pickle("/Users/zihengchen/Documents/BRIL/data/TBTree/{}.pkl".format(run))
    
    processes = []
    for config in [("198","23"), ("231","01") ]:
        processes.append(mp.Process(target=processFile,args=(config,)))
    for ps in processes:
        ps.start()
    for ps in processes:
        ps.join()
    