from tb_DataReader import EventFeaturesReader
from tb_PulseAmpFitter import PulseAmpFitter
import pandas as pd
from pylab import *
import seaborn


class HVAnalyzer():
  def __init__(self,dataDir,runMngs):
    self.dataDir = dataDir
    self.runMngs = runMngs

  def fitAndPlotRuns(self, makePlots=True, plotDir="../plots/pulseAmpFit"):
    saveVariables = ["mu","mu_err","bs_mean"]

    for runMng in self.runMngs:
      sns = runMng["sns"]
      channels = runMng["channels"]
      for irun,run in enumerate(runMng["run"]):
        u = runMng["u"][irun]
        for backend in ["ubcm","VME"]:

          # if not available
          if runMng[backend+"_fitxmin"][0,irun] == -1:
            for channel in channels: 
              for p in ["mu","mu_err","argmax"]:
                key = "{}_{}_CH{}".format(p,backend,channel)
                if key in runMng:
                  runMng[key].append(np.nan)
                else:
                  runMng[key] = [np.nan]  
            
              for p in ["baseline_mean"]:
                key = "{}_{}_CH{}".format(p,backend,channel)
                if key in runMng:
                  runMng[key].append(np.nan)
                else:
                  runMng[key] = [np.nan]

            continue

          # if available
          rd = EventFeaturesReader(self.dataDir,"000"+run,backend)
          for ich,channel in enumerate(channels):  
            ch = rd.getChannel(channel)
            sigma = ch.baseline_std.mean()
            title = "Run {}, sensor-{} {}V, {} CH{}, n={}".format(run,sns, u,backend,channel,len(ch))
            # fitting
            fitxmin = runMng[backend+"_fitxmin"][ich,irun]
            fitxmax = runMng[backend+"_fitxmax"][ich,irun]
            fitter = PulseAmpFitter(ch.pulseAmp)
            fitter.fitRange(fitxmin,fitxmax,None)

            # plot
            if makePlots:
              fitter.plot()
              plt.title(title, fontsize=14)
              plt.savefig(plotDir+"/langaus_{}_run{}_{}.png".format(backend,run,channel),dpi=300)
              plt.close()

            # save fitting result
            for p in ["mu","mu_err","argmax"]:
              key = "{}_{}_CH{}".format(p,backend,channel)
              if key in runMng:
                runMng[key].append(fitter.param[p])
              else:
                runMng[key] = [fitter.param[p]]

            # save other info
            for p in ["baseline_mean"]:
              key = "{}_{}_CH{}".format(p,backend,channel)
              if key in runMng:
                runMng[key].append(ch[p].mean())
              else:
                runMng[key] = [ch[p].mean()]


    return self.runMngs
  
    

  def plotRuns_Hist2D(self, plotDir="../plots/PulseHist2D"):
    for runMng in self.runMngs:
      sns = runMng["sns"]
      channels = runMng["channels"]
      for irun,run in enumerate(runMng["run"]):
        u = runMng["u"][irun]
        for backend in ["ubcm","VME"]:

          # if not available
          if runMng[backend+"_fitxmin"][0,irun] == -1:
            continue

          # if available
          rd = EventFeaturesReader(self.dataDir,"000"+run,backend)
          for ich,channel in enumerate(channels):  
            ch = rd.getChannel(channel)
            title = "Run {}, sensor-{} {}V, {} CH{}, n={}".format(run,sns, u,backend,channel,len(ch))
            
            seaborn.set(style="ticks")
            jp= seaborn.jointplot(ch.pulseAmp_sigmaNoise, ch.pulsePos, kind="hex", color="C0",gridsize=30,ratio=4,space=0.5)
            jp.set_axis_labels(r'Amplitude [baseline $\sigma$]','Position')
            jp.ax_joint.set_title(title)
            plt.savefig(plotDir+"/{}_run{}_{}.png".format(backend,run,channel),dpi=300)
            plt.close()