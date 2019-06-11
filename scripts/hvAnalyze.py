#!/usr/bin/env python

import pandas as pd
from pylab import *
from tb_DataReader import EventFeaturesReader
from tb_PulseAmpFitter import PulseAmpFitter
from tb_RunManager import getRunMngs
from tb_HVAnalyzer import HVAnalyzer
from IPython.display import clear_output


dataDir = "../data/eventFeatures"
runMngs = getRunMngs()
ana = HVAnalyzer(dataDir,runMngs)
runMngs = ana.fitAndPlotRuns(makePlots=False)




def plotHVScan():

  f, (ax1, ax2) = plt.subplots(2,1, sharey=True, facecolor='w',figsize=(8,6))
  plt.subplots_adjust(hspace=0)
  #plotParam = "mu"
  plotParam = "argmax"

  runMng = runMngs[0]
  l = "ubcm_CH2"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax1.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='^-',color='C0',label='sensor1 '+l)

  l = "ubcm_CH3"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax1.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='o--',color='C0',label='sensor1 '+l)

  l = "VME_CH2"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax1.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='^-',color='C1',label='sensor1 '+l)

  l = "VME_CH3"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax1.errorbar(u[~np.isnan(y)], y[~np.isnan(y)],  fmt='o--',color='C1',label='sensor1 '+l)


  ax1.set_xlim(0,1200)
  ax1.set_ylim(0,70)
  ax1.legend(loc="upper right")
  ax1.grid(linestyle='--',alpha=0.3)
  ax1.set_ylabel("Amplitude MPV")
  ax1.set_xticklabels([])


  runMng = runMngs[1]
  l = "ubcm_CH0"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax2.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='^-',color='C2',label='sensor2 '+l)

  l = "ubcm_CH1"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax2.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='o--',color='C2',label='sensor2 '+l)

  l = "VME_CH0"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax2.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='^-',color='C3',label='sensor2 '+l)

  l = "VME_CH1"
  u, y = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l])
  ax2.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='o--',color='C3',label='sensor2 '+l)

  ax2.set_xlim(0,1200)
  ax2.set_ylim(0,70)


  ax2.legend(loc="upper right")
  ax2.grid(linestyle='--',alpha=0.3)
  ax2.set_xlabel("Voltage [V]")
  ax2.set_ylabel("Amplitude MPV")


def plotHVScanWithCalibration():
  const = pd.read_pickle("../data/calibration_linearFit.pkl")
  plt.figure(facecolor='w',figsize=(7,4))
  plotParam = "argmax"

  runMng = runMngs[0]
  l = "ubcm_CH2"
  u, y, y_bs = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l]), np.array(runMng['baseline_mean_'+l])
  y = y_bs + y
  y = (y*const[l][0] + const[l][1])/80
  plt.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='^-',color='C0',label='sensor1 '+l)

  l = "ubcm_CH3"
  u, y, y_bs = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l]), np.array(runMng['baseline_mean_'+l])
  y = y_bs + y
  y = (y*const[l][0] + const[l][1])/80
  plt.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='o--',color='C0',label='sensor1 '+l)

  l = "VME_CH2"
  u, y, y_bs = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l]), np.array(runMng['baseline_mean_'+l])
  y = y_bs - y
  y = (y*const[l][0] + const[l][1])/80
  plt.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='^-',color='C1',label='sensor1 '+l)

  l = "VME_CH3"
  u, y, y_bs = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l]), np.array(runMng['baseline_mean_'+l])
  y = y_bs - y
  y = (y*const[l][0] + const[l][1])/80
  plt.errorbar(u[~np.isnan(y)], y[~np.isnan(y)],  fmt='o--',color='C1',label='sensor1 '+l)



  runMng = runMngs[1]

  l = "ubcm_CH1"
  u, y, y_bs = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l]), np.array(runMng['baseline_mean_'+l])
  y = y_bs + y
  y = (y*const[l][0] + const[l][1])/80
  plt.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='o--',color='C2',label='sensor2 '+l)

  l = "VME_CH1"
  u, y, y_bs = np.array(runMng['u']), np.array(runMng[plotParam+'_'+l]), np.array(runMng['baseline_mean_'+l])
  y = y_bs - y
  y = (y*const[l][0] + const[l][1])/80
  plt.errorbar(u[~np.isnan(y)], y[~np.isnan(y)], fmt='o--',color='C3',label='sensor2 '+l)

  plt.xlim(0,900)
  plt.ylim(0,300)


  plt.legend(loc="lower right")
  plt.grid(linestyle='--',alpha=0.3)
  plt.ylabel("Effective Thickness [um]")
  plt.xlabel("Voltage [V]")
  plt.text(100,40,"Extracted from LanGau Fit with")
  plt.text(100,20,"test pulse calibration and 80 e/um")

if __name__=="__main__":
  ana.plotRuns_Hist2D()
  plotHVScan()
  plotHVScanWithCalibration()
  plt.show()
  