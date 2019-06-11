
from pylab import *
import ROOT
from langaus.langaus import LanGausFit


class PulseAmpFitter():
  def __init__(self, amp, xmin=0, xmax=120, binsize=1):

    self.amp = amp
    self.xmin = xmin
    self.xmax = xmax
    self.binsize = binsize
    self.mybins = np.arange(xmin,xmax+binsize,binsize)
    self.centers = self.mybins[:-1]

  def fitRange(self, fit_xmim, fit_xmax, gaussianSigma):
    self.fit_xmim = fit_xmim
    self.fit_xmax = fit_xmax

    self.h,_ = np.histogram(self.amp, bins=self.mybins)
    # fill root histogram
    histogram = ROOT.TH1D("hist", "hist", self.mybins.size, self.xmin, self.xmax)
    for i in range(self.centers.size):
      histogram.Fill(self.centers[i],self.h[i])

    # for x in np.arange(self.fit_xmim, self.fit_xmax, self.binsize):
    #     i = int((x-self.xmin)/self.binsize)
    #     histogram.Fill(self.centers[i],self.h[i])
    try:
      # fit root histogram
      self.func = LanGausFit().fit(histogram, fitrange=(fit_xmim,fit_xmax), startsigma=gaussianSigma)
      self.getParem()
    except:
      print("fitting error")
  
  def getParem(self):
    # save parameter
    self.param = {}
    self.param["c"]     = self.func.GetParameter(0)
    self.param["mu"]    = self.func.GetParameter(1)
    self.param["norm"]  = self.func.GetParameter(2)
    self.param["sigma"] = self.func.GetParameter(3)
    self.param["c_err"]     = self.func.GetParError(0)
    self.param["mu_err"]    = self.func.GetParError(1)    
    self.param["norm_err"]  = self.func.GetParError(2)
    self.param["sigma_err"] = self.func.GetParError(3)


    maxLangau = 0
    argmaxLangau = self.param["c"]
    for x in np.arange(self.fit_xmim, self.fit_xmax,0.01):
      temp = self.func(x)
      if temp > maxLangau:
        argmaxLangau = x
        maxLangau =  temp
    self.param["argmax"] = argmaxLangau

  def plot(self):
    plt.figure(facecolor='w',figsize=(10,4))
    
    # plot fittingp
    xarray = np.arange(self.xmin,self.xmax,0.5)
    laugau = [self.func(x) for x in xarray]
    plt.axvspan(self.fit_xmim, self.fit_xmax, lw=0, alpha=0.1, color='C0')
    plt.plot(xarray,laugau,lw=3,color='C0',label='LanGau Fit')
    
    # plot data
    plt.errorbar( self.centers, self.h, yerr=sqrt(self.h), xerr=self.binsize,
                  fmt='.',color='k',label='Testbeam Data')

    
    plt.grid(linestyle='--',alpha=0.3)
    plt.legend(fontsize=14)
    plt.xlabel("Pluse Amplitude", fontsize=12)
    norm = 1.5*np.array(laugau).max()
    plt.ylim(0,norm)

    xtxt,ytxt,ytxtspace = 0.7*self.xmax, 0.3*norm, 0.1*norm
    plt.text(xtxt,ytxt+2*ytxtspace,r"Landau MPV={:>6.3f}$\pm${:>6.3f}".format(self.param["mu"],self.param["mu_err"]))
    plt.text(xtxt,ytxt+ytxtspace,r"Landau Width={:>6.3f}$\pm${:>6.3f}".format(self.param["c"],self.param["c_err"]))
    plt.text(xtxt,ytxt,r"Gaussian Width={:>6.3f}$\pm${:>6.3f}".format(self.param["sigma"],self.param["sigma_err"]))
    
    plt.axvline(self.param["argmax"],color='C0',linestyle='--')
    plt.text( self.param["argmax"], 0.05*norm, '{:>7.2f}'.format(self.param["argmax"]), color = "C0")
  