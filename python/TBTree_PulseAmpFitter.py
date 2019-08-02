
from pylab import *
import ROOT
from langaus.langaus import LanGausFit
from scipy.interpolate import UnivariateSpline


class PulseAmpFitter():
  def __init__(self, amp, xmin=0, xmax=120, binsize=1):

    self.amp = amp
    self.xmin = xmin
    self.xmax = xmax
    self.binsize = binsize
    self.mybins = np.arange(xmin,xmax+binsize,binsize)
    self.centers = self.mybins[:-1]+binsize/2

  def adjustFitRange(self, fitRangeFinder_xmin, fitRangeFinder_xmax, fractionOfMax_left=0.3, fractionOfMax_right=0.2 ):


      xLeft, xRight = fitRangeFinder_xmin, fitRangeFinder_xmax

      x = self.centers
      h,_ = np.histogram(self.amp, bins=self.mybins)
      inRange = np.logical_and( (x>fitRangeFinder_xmin), (x<fitRangeFinder_xmax))
      x = x[inRange]
      h = h[inRange]

      hMax = np.max(h)
      mode = x[np.argmax(h)] 

      xLeftFound = False
      for xx,hh in zip(x, h):
        if (xx<mode and hh>= fractionOfMax_left*hMax and xLeftFound==False):
          xLeft = xx
          xLeftFound =True
        if (xx>mode and hh>= fractionOfMax_right*hMax ):
          xRight = xx

      
      return xLeft, xRight



  def fitRange(self, fit_xmin, fit_xmax, gaussianSigma, adjustFitRange=False  ):
    self.fit_xmin_0, self.fit_xmax_0  = fit_xmin, fit_xmax

    if adjustFitRange:
      self.fit_xmin, self.fit_xmax  = self.adjustFitRange(fit_xmin, fit_xmax)
      self.alpha_fitRange0 = 0.1
    else:
      self.fit_xmin, self.fit_xmax  = self.fit_xmin_0, self.fit_xmax_0 
      self.alpha_fitRange0 = 0.0

    self.h,_ = np.histogram(self.amp, bins=self.mybins)
    # fill root histogram
    histogram = ROOT.TH1D("hist", "hist", self.mybins.size, self.xmin, self.xmax)
    for i in range(self.centers.size):
      if (self.centers[i]>=self.fit_xmin and self.centers[i]<=self.fit_xmax ):
        histogram.Fill(self.centers[i],self.h[i])

    try:
      # fit root histogram
      self.func = LanGausFit().fit(histogram, fitrange=(self.fit_xmin,self.fit_xmax), startsigma=gaussianSigma)
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
    for x in np.arange(self.fit_xmin, self.fit_xmax,0.01):
      temp = self.func(x)
      if temp > maxLangau:
        argmaxLangau = x
        maxLangau =  temp
    self.param["argmax"] = argmaxLangau

  def plot(self):
    plt.figure()
    fig, axes = plt.subplots(2, 1, sharex=True, 
                              gridspec_kw={'height_ratios':[3,1]},
                              facecolor='w',figsize=(8,5))
    fig.subplots_adjust(hspace=0)

    self.axes = axes

    ax = axes[0]    
    # plot fittingp
    xarray = np.arange(self.xmin,self.xmax,0.5)
    laugau = [self.func(x) for x in xarray]
    
    ax.plot(xarray,laugau,lw=3,color='C0',label='LanGau Fit')
    
    # plot data
    ax.errorbar(self.centers, self.h, yerr=sqrt(self.h), xerr=self.binsize/2,
                fmt='.',color='k',label='Data')

    norm = 1.5*np.array(laugau).max()
    ax.set_ylim(0,norm)
    xtxt,ytxt,ytxtspace = 0.63*self.xmax, 0.3*norm, 0.1*norm
    ax.text(xtxt,ytxt+2*ytxtspace,r"Landau MPV={:>6.3f}$\pm${:>6.3f}".format(self.param["mu"],self.param["mu_err"]))
    ax.text(xtxt,ytxt+ytxtspace,r"Landau Width={:>6.3f}$\pm${:>6.3f}".format(self.param["c"],self.param["c_err"]))
    ax.text(xtxt,ytxt,r"Gaussian Width={:>6.3f}$\pm${:>6.3f}".format(self.param["sigma"],self.param["sigma_err"]))
    
    
    ax.axvspan(self.fit_xmin_0, self.fit_xmax_0, lw=0, alpha=self.alpha_fitRange0, color='gray')
    ax.axvspan(self.fit_xmin, self.fit_xmax, lw=0, alpha=0.1, color='C0',label="Fitting Range")

    ax.axvline(self.param["argmax"],color='C0',linestyle='--')
    ax.text( self.param["argmax"], 0.05*norm, '{:>7.2f}'.format(self.param["argmax"]), color = "C0")
    #ax.plot( [self.fit_xmin, self.param["argmax"]], [self.hThresholdLeft,self.hThresholdLeft], color='C0',linestyle=':')
    #ax.plot( [self.fit_xmax, self.param["argmax"]], [self.hThresholdRight,self.hThresholdRight], color='C0',linestyle=':')
    
    ax.grid(linestyle='--',alpha=0.3)
    ax.legend(fontsize=14)
    
    
  
    ax = axes[1]

    landau = np.array([self.func(x) for x in self.centers]) 
    ratio = (self.h-landau)/landau
    ratio_err = self.h**0.5/landau

    slt = (self.centers>self.fit_xmin) & (self.centers<self.fit_xmax)

    ax.errorbar(self.centers[slt], ratio[slt], yerr=ratio_err[slt], xerr=self.binsize/2, fmt='.',color='k')
    ax.axhline(0, lw=3, color = "C0")
    
    ax.axvspan(self.fit_xmin_0, self.fit_xmax_0, lw=0, alpha=self.alpha_fitRange0, color='gray')
    ax.axvspan(self.fit_xmin, self.fit_xmax, lw=0, alpha=0.1, color='C0')

    ax.axvline(self.param["argmax"],color='C0',linestyle='--')

    ax.grid(linestyle='--',alpha=0.3)
    ax.set_ylim(-0.5,0.5)
    ax.set_ylabel("(N-f)/f")
    ax.set_xlim(self.xmin,self.xmax)
    ax.set_xlabel("Pluse Amplitude", fontsize=12)

