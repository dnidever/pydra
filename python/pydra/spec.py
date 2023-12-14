#!/usr/env python

"""SPEC.PY - Spectroscopy tools

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd

# Imports

import os
import numpy as np
import copy
import time
import warnings
from astropy.io import fits
from astropy.table import Table
from dlnpyutils.minpack import curve_fit
from dlnpyutils import utils as dln, bindata, robust
from numpy.polynomial.polynomial import polyfit as npp_polyfit, polyval as npp_polyval
from scipy import ndimage
from scipy.signal import medfilt, argrelextrema
from scipy.ndimage.filters import median_filter,gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf, wofz
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

def gaussbin(x, amp, cen, sig, const=0, dx=1.0):
    """1-D gaussian with pixel binning
    
    This function returns a binned Gaussian
    par = [height, center, sigma]
    
    Parameters
    ----------
    x : array
       The array of X-values.
    amp : float
       The Gaussian height/amplitude.
    cen : float
       The central position of the Gaussian.
    sig : float
       The Gaussian sigma.
    const : float, optional, default=0.0
       A constant offset.
    dx : float, optional, default=1.0
      The width of each "pixel" (scalar).
    
    Returns
    -------
    geval : array
          The binned Gaussian in the pixel

    """

    xcen = np.array(x)-cen             # relative to the center
    x1cen = xcen - 0.5*dx  # left side of bin
    x2cen = xcen + 0.5*dx  # right side of bin

    t1cen = x1cen/(np.sqrt(2.0)*sig)  # scale to a unitless Gaussian
    t2cen = x2cen/(np.sqrt(2.0)*sig)

    # For each value we need to calculate two integrals
    #  one on the left side and one on the right side

    # Evaluate each point
    #   ERF = 2/sqrt(pi) * Integral(t=0-z) exp(-t^2) dt
    #   negative for negative z
    geval_lower = erf(t1cen)
    geval_upper = erf(t2cen)

    geval = amp*np.sqrt(2.0)*sig * np.sqrt(np.pi)/2.0 * ( geval_upper - geval_lower )
    geval += const   # add constant offset

    return geval


def gaussian(x, amp, cen, sig, const=0):
    """1-D gaussian: gaussian(x, amp, cen, sig)"""
    return amp * np.exp(-(x-cen)**2 / (2*sig**2)) + const


def gaussfit(x,y,initpar=None,sigma=None,bounds=(-np.inf,np.inf),binned=False):
    """Fit a Gaussian to data."""
    if initpar is None:
        initpar = [np.max(y),x[np.argmax(y)],1.0,np.median(y)]
    func = gaussian
    if binned is True: func=gaussbin
    return curve_fit(func, x, y, p0=initpar, sigma=sigma, bounds=bounds)



def wavesol(xpix,wave,order=3,xr=None):
    """
    Fit wavelength solution to X and Wavelength arrays.

    """

    n = len(xpix)
    if n<2:
        raise ValueError('Need at least two points.')
    if n<order+1:
        print('Warning: order='+str(order)+' but only '+str(n)+' points.  Reducing to order='+str(n-1))
        order = n-1

    # Robust polynomial fit
    coef = robust.polyfit(xpix,wave,order)

    # Generate output array of wavelength values
    if xr is None:
        xx = np.arange(np.floor(np.min(xpix)),np.ceil(np.max(xpix)))

    else:
        xx = np.arange(xr[0],xr[1])
    ww = npp_polyval(xx,coef)        

    return coef,ww
        

def trace(im,yestimate=None,yorder=2,sigorder=4,step=10,spectral_axis=1):
    """
    Trace the spectrum.  Spectral dimension is assumed to be on the horizontal axis.

    Parameters
    ----------
    im : numpy array
       The input 2D image.
    yestimate : float, optional
       The initial estimate of the central Y (spatial dimension) position of the trace.  Default
          is to calculate using a median cut.
    yorder : int, optional
       Polynomial order to use for fitting the trace position as a function of column.  Default is 2.
    sigorder : int, optional
       Polynomial order to use for fitting the Gaussian sigma as a function of column.  Default is 4.
    step : int, optional
       Stepsize to take in the spectral dimension when tracing the spectrum.  Default is 10 pixels.
    spectral_axis : int, optional
       The spectral axis.  Default is 1.

    Returns
    -------
    tcat : table
       Table of Gaussian fits when stepping along in columns and tracing the spectrum.
    ypars : numpy array
       Polynomial coefficients of the trace.
    sigpars : numpy array
       Polynomial coefficients of the Gaussian sigma.
    mcat : table
       Table of model x, y and sigma values along the spectrum.

    Example
    -------

    tcat,ypars,sigpars,mcat = trace(im)    

    """

    spec = np.copy(im)  # internal copy
    if spectral_axis==0:  # transpose
        spec = spec.T
    ny,nx = spec.shape
    y = np.arange(ny)
    if yestimate is None:
        ytot = np.sum(im,axis=1)
        yestimate = np.argmax(ytot)
    # Smooth in spectral dimension
    # a uniform (boxcar) filter with a width of 50
    smim = ndimage.uniform_filter1d(im, 50, 1)
    nstep = nx//step
    # Loop over the columns in steps and fit Gaussians
    tcat = np.zeros(nstep,dtype=np.dtype([('x',float),('amp',float),('y',float),('sigma',float),
                                          ('pars',float,4),('perror',float,4)]))
    for i in range(nstep):
        pars,cov = dln.gaussfit(y[yestimate-10:yestimate+10],im[yestimate-10:yestimate+10,step*i+step//2])
        perror = np.sqrt(np.diag(cov))
        tcat['x'][i] = step*i+step//2
        tcat['amp'][i] = pars[0]
        tcat['y'][i] = pars[1]
        tcat['sigma'][i] = pars[2]
        tcat['pars'][i] = pars
        tcat['perror'][i] = perror        
    # Fit polynomail to y vs. x and gaussian sigma vs. x
    ypars = np.polyfit(tcat['x'],tcat['pars'][:,1],yorder)
    sigpars = np.polyfit(tcat['x'],tcat['pars'][:,2],sigorder)
    # Model
    mcat = np.zeros(nx,dtype=np.dtype([('x',float),('y',float),('sigma',float)]))
    xx = np.arange(nx)
    mcat['x'] = xx
    mcat['y'] = np.poly1d(ypars)(xx)
    mcat['sigma'] = np.poly1d(sigpars)(xx)
    return tcat, ypars, sigpars,mcat

def boxsum(im,ylo,yhi):
    """ Helper function for boxcar extraction."""
    y0 = np.min(ylo).astype(int)
    y1 = np.max(yhi).astype(int)
    # Sum up the flux
    subim = im[y0:y1,:]
    ny,nx = subim.shape
    xx,yy = np.meshgrid(np.arange(nx),np.arange(ny))
    mask = (yy>=(ylo-y0)) & (yy<=(yhi-y0))
    flux = np.sum(mask*subim,axis=0)
    return flux
    
    
def boxcar(im,ytrace=None,width=20,backlo=None,backhi=None):
    """
    Boxcar extract the spectrum

    Parameters
    ----------
    im : numpy array
      Image from which to extract the spectrum.
    ytrace : numpy array, optional
      The y (spatial) position of the trace as a function of column.
    width : int, optional
      The half-width of the spectrum in the spatial dimension to extract.
    backlo : tuple, optional
       The lower and upper offsets (relative to the trace) for the lower
         background region (e.g., (-50,-40)).  Default is None.
    backhi : tuple, optional
       The lower and upper offsets (relative to the trace) for the upper.
         background region (e.g., (40,50)).  Default is None.

    Returns
    -------
    out : table
       Output table with flux and background values.

    Example
    -------

    out = boxcar(im)

    """
    
    ny,nx = im.shape
    # Get median trace position
    if ytrace is None:
        ytot = np.sum(im,axis=1)
        yest = np.argmax(ytot)
        ytrace = np.zeros(nx,float)+yest
        
    # Start output
    dt = np.dtype([('x',float),('ytrace',float),('sumflux',float),('background',float),
                 ('backlo',float),('backhi',float),('flux',float)])
    out = np.zeros(nx,dtype=dt)
    out['x'] = np.arange(nx)
    out['ytrace'] = ytrace
    
    # Sum up the flux
    ylo = np.maximum(ytrace-width,0).astype(int)
    yhi = np.minimum(ytrace+width,ny).astype(int)
    flux = boxsum(im,ylo,yhi)

    out['sumflux'] = flux
    # Background
    bflux = None
    if backlo is not None:
        bloylo = np.maximum(ytrace+backlo[0],0).astype(int)
        bloyhi = np.maximum(ytrace+backlo[1],0).astype(int)
        bloflux = boxsum(im,bloylo,bloyhi)
        bflux = bloflux
        out['backlo'] = bloflux
    if backhi is not None:
        bhiylo = np.minimum(ytrace+backhi[0],ny).astype(int)
        bhiyhi = np.minimum(ytrace+backhi[1],ny).astype(int)
        bhiflux = boxsum(im,bhiylo,bhiyhi)
        bflux = bhiflux
        out['backhi'] = bhiflux        
    # Average backgrounds
    if backlo is not None and backhi is not None:
       bflux = 0.5*(bloflux+bhiflux) 

    # Final flux
    if bflux is not None:
        out['background'] = bflux
        out['flux'] = flux-bflux
    else:
        out['flux'] = flux        
        
    return out


def gaussline(x, amp, cen, sigma, const=0):
    """1-D gaussian: gaussian(x, amp, cen, sig)"""
    return amp * np.exp(-(x-cen)**2 / (2*sigma**2)) + const

def fitlines(x,y,err=None,nsig=5):
    """
    Automatically detect emission lines and fit Gaussians to them.
    """

    n = len(x)
    sm = 101
    
    # Detect peaks
    med = medfilt(y,sm)
    sig = dln.mad(y-med,zero=True)
    peaks, = argrelextrema(y,np.greater)
    good, = np.where((y[peaks]-med[peaks] > nsig*sig) & (y[peaks]>0))
    ngood = len(good)
    if ngood==0:
        print('No peaks found')
        return None
    peaks = peaks[good]
    
    # Loop over peaks and fit Gaussians
    dt = np.dtype([('id',int),('xpeak',int),('amp',float),('xcen',float),('sigma',float),('const',float),
                   ('gpar',(float,4)),('gparerr',(float,4)),('flux',float),('success',bool)])
    out = np.zeros(ngood,dtype=dt)
    out['id'] = np.arange(ngood)+1
    for i in range(ngood):
        x0 = np.maximum(peaks[i]-15,0)
        x1 = np.minimum(peaks[i]+15,n-1)
        if err is not None:
            err1 = err[x0:x1+1]
        else:
            err1 = None
        initpar = [y[peaks[i]]-med[peaks[i]],peaks[i],5.0,med[peaks[i]]]
        lbounds = np.zeros(4,float)
        lbounds[1] = peaks[i]-2
        lbounds[3] = np.min(y)
        ubounds = np.zeros(4,float)
        ubounds[0] = np.inf
        ubounds[1] = peaks[i]+2
        ubounds[2] = np.inf
        ubounds[3] = np.max(y)
        bounds = (lbounds,ubounds)
        out['xpeak'][i] = peaks[i]
        try:
            par1, cov1 = curve_fit(gaussline, x[x0:x1+1], y[x0:x1+1], p0=initpar, bounds=bounds, sigma=err1)
            perror1 = np.sqrt(np.diag(cov1))
            flux1 = par1[0]*par1[2]*np.sqrt(2*np.pi)
            out['gpar'][i] = par1
            out['gparerr'][i] = perror1
            out['amp'][i] = par1[0]
            out['xcen'][i] = par1[1]
            out['sigma'][i] = par1[2]
            out['const'][i] = par1[3]            
            out['flux'][i] = flux1
            out['success'][i] = True
        except:
            out['amp'][i] = initpar[0]
            out['xcen'][i] = initpar[1]
            out['sigma'][i] = initpar[2]
            out['const'][i] = initpar[3]  
            out['success'][i] = False        
        
    return out
        
    
def linefit(x,y,initpar,bounds,err=None):
    # Fit Gaussian profile to data with center and sigma fixed.
    # for extracting spectra
    # initpar = [height, center, sigma, constant offset]
    cen = initpar[1]
    sigma = initpar[2]
    def gline(x, amp, const=0):
        """1-D gaussian: gaussian(x, amp, cen, sig)"""
        return amp * np.exp(-(x-cen)**2 / (2*sigma**2)) + const
    line_initpar = [initpar[0],initpar[3]]
    lbounds, ubounds = bounds
    line_bounds = ([lbounds[0],lbounds[3]],[ubounds[0],ubounds[3]])
    return curve_fit(gline, x, y, p0=line_initpar, bounds=line_bounds, sigma=err)


def extract(im,imerr=None,mcat=None,nobackground=False,verbose=False):
    """
    Extract a spectrum

    Parameters
    ----------
    im : numpy array
       The 2D image array from which to extract the spectra
    imerr : numpy array, optional
       Uncertainty array for im.
    mcat : table
       Table containing the trace information returned from trace().
         This must (at last) contain the "y" and "sigma" columns.
         If this is not input, then trace() will be run on the image.
    nobackground : boolean, optional
       Do not subtract the background.  This is useful when extracting
         arc lamp data from longslit spectra.  Default is False.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    cat : table
       Table containing the extracted spectrum and uncertainties.

    Example
    -------

    cat = extract(im,imerr,mcat)

    """
    
    ny,nx = im.shape
    x = np.arange(nx)
    y = np.arange(ny)
    # No trace information input, get it
    if mcat is None:
        tcat,ypars,sigpars,mcat = trace(im)
        
    # Initialize the output
    cat = np.zeros(nx,dtype=np.dtype([('x',int),('pars',float,2),('perr',float,2),
                                      ('flux',float),('fluxerr',float)]))
    # Loop over the columns and get the flux using the trace information
    for i in range(nx):
        line = im[:,i].flatten()
        if imerr is not None:
            lineerr = imerr[:,i].flatten()
        else:
            lineerr = np.ones(len(line))   # unweighted
        # Fit the constant offset and the height of the Gaussian
        #  fix the central position and sigma
        ycen = mcat['y'][i]
        ysigma = mcat['sigma'][i]
        ht0 = np.maximum(line[int(np.round(ycen))],0.01)
        initpar = [ht0,ycen,ysigma,np.median(line)]
        if nobackground is True:
            initpar = [ht0,ycen,ysigma,0]
        # Only fit the region right around the peak
        y0 = int(np.maximum(ycen-50,0))
        y1 = int(np.minimum(ycen+50,ny))
        bnds = ([0,ycen-1e-7,ysigma-1e-7,np.min(line)],[1.5*ht0,ycen+1e-7,ysigma+1e-7,initpar[3]+3*dln.mad(line)])
        if nobackground is True:
            initpar[3] = 0.0
            bnds = ([0,ycen-1e-4,ysigma-1e-4,-1e-7],[1.5*ht0,ycen,ysigma,1e-7])
        # We could probably use linear algebra to figure this out more quickly
        pars,cov = linefit(y[y0:y1],line[y0:y1],initpar=initpar,bounds=bnds,err=lineerr[y0:y1])
        perr = np.sqrt(np.diag(cov))
        # Gaussian area = ht*wid*sqrt(2*pi)
        flux = pars[0]*ysigma*np.sqrt(2*np.pi)
        fluxerr = perr[0]*ysigma*np.sqrt(2*np.pi)
        cat['x'][i] = i
        cat['pars'][i] = pars
        cat['perr'][i] = perr
        cat['flux'][i] = flux
        cat['fluxerr'][i] = fluxerr
        if verbose:
            print(i,pars)
    return cat

def emissionlines(spec,thresh=None):
    """
    Measure the emission lines in an arc lamp spectrum.

    Parameters
    ----------
    spec : numpy array
       The input spectrum in which to detect emission peaks.
    thresh : float, optional
       The threshold to use for detecting peaks.  The default
          is 5 sigma above the background.

    Returns
    -------
    lines : table
        Output table with measurements of peaks.
    model : numpy array
        Model spectrum of the peaks.

    Example
    -------

    lines, model = spec.emissionlines(flux)

    """
    nx = len(spec)
    x = np.arange(nx)

    # Remove background
    med = medfilt(spec,101)
    subspec = spec-med
    
    # Threshold
    if thresh is None:
        #thresh = (np.max(subspec)-np.min(subspec))*0.05
        sig = dln.mad(subspec)
        thresh = 5*sig
        
    # Detect the peaks
    sleft = np.hstack((0,subspec[0:-1]))
    sright = np.hstack((subspec[1:],0))
    peaks, = np.where((subspec>sleft) & (subspec>sright) & (subspec>thresh))
    npeaks = len(peaks)
    print(str(npeaks)+' peaks found')
    
    # Loop over the peaks and fit them with Gaussians
    gcat = np.zeros(npeaks,dtype=np.dtype([('x0',int),('x',float),('xerr',float),('pars',float,4),('perr',float,4),
                                           ('flux',float),('fluxerr',float),('amp',float),('sigma',float),('success',bool)]))
    resid = subspec.copy()
    gmodel = np.zeros(nx)
    for i in range(npeaks):
        x0 = peaks[i]
        xlo = np.maximum(x0-6,0)
        xhi = np.minimum(x0+6,nx)
        initpar = [subspec[x0],x0,1,0]
        bnds = ([0,x0-3,0.1,0],[1.5*initpar[0],x0+3,10,1e4])
        try:
            pars,cov = dln.gaussfit(x[xlo:xhi],subspec[xlo:xhi],initpar,bounds=bnds,binned=True)
            perr = np.sqrt(np.diag(cov))
            gcat['success'][i] = True
        except:
            pars = initpar
            perr = np.copy(initpar)*0.0
            gcat['success'][i] = False            
        gmodel1 = dln.gaussian(x[xlo:xhi],*pars)
        gmodel[xlo:xhi] += (gmodel1-pars[3])
        resid[xlo:xhi] -= (gmodel1-pars[3])
        # Gaussian area = ht*wid*sqrt(2*pi)
        flux = pars[0]*pars[2]*np.sqrt(2*np.pi)
        fluxerr = perr[0]*pars[2]*np.sqrt(2*np.pi)
        gcat['x0'][i] = x0
        gcat['x'][i] = pars[1]
        gcat['xerr'][i] = perr[1]
        gcat['pars'][i] = pars
        gcat['perr'][i] = perr
        gcat['flux'][i] = flux
        gcat['fluxerr'][i] = fluxerr
        gcat['amp'][i] = pars[0]
        gcat['sigma'][i] = pars[2]
        
    return gcat, gmodel

def matchlines(w1,w2,dcr=0.5):
    """
    Match up two lists of lines within a 0.5 toleration.
    
    Parameters
    ----------
    w1 : numpy array
       First array of wavelengths.
    w2 : numpy array
       Second array of wavelengths.
    dcr : float
       The matching radius.

    Returns
    -------
    ind1 : numpy array
       Array of indices into w1 for matches.  None if no matches.
    ind2 : numpy array
       Array of indices into w2 for matches.  None if no matches.

    Example
    -------
    
    ind1,ind2 = matchlines(w1,w2,2.0)

    """
    match = np.zeros(len(w1),bool)
    wdiff = np.zeros(len(w1),float)
    wmatch = np.zeros(len(w1),np.float64)
    xmatch = np.zeros(len(w1),float)
    indmatch = np.zeros(len(w1),int)-1
    # Loop over w2
    for i in range(len(w1)):
        dist = np.abs(w1[i]-w2)
        mindist = np.min(dist)
        ind = np.argmin(dist)
        if mindist<dcr:
            match[i] = True
            indmatch[i] = ind
    ind1, = np.where(match==True)
    print(len(ind1),' matches')
    if len(ind1)>0:
        ind2 = indmatch[ind1]
    else:
        ind1,ind2 = None,None
        
    return ind1,ind2

def continuum(spec,bin=50,perc=60,norder=4):
    """ Derive the continuum of a spectrum."""
    nx = len(spec)
    x = np.arange(nx)
    # Loop over bins and find the maximum
    nbins = nx//bin
    xbin1 = np.zeros(nbins,float)
    ybin1 = np.zeros(nbins,float)
    for i in range(nbins):
        xbin1[i] = np.mean(x[i*bin:i*bin+bin])
        ybin1[i] = np.percentile(spec[i*bin:i*bin+bin],perc)
    # Fit polynomial to the binned values
    coef1 = np.polyfit(xbin1,ybin1,norder)
    cont1 = np.poly1d(coef1)(x)
    
    # Now remove large negative outliers and refit
    gdmask = np.zeros(nx,bool)
    gdmask[(spec/cont1)>0.8] = True
    xbin = np.zeros(nbins,float)
    ybin = np.zeros(nbins,float)
    for i in range(nbins):
        xbin[i] = np.mean(x[i*bin:i*bin+bin][gdmask[i*bin:i*bin+bin]])
        ybin[i] = np.percentile(spec[i*bin:i*bin+bin][gdmask[i*bin:i*bin+bin]],perc)
    # Fit polynomial to the binned values
    coef = np.polyfit(xbin,ybin,norder)
    cont = np.poly1d(coef)(x)
    
    return cont,coef

def voigt(x, height, cen, sigma, gamma, const=0.0, slp=0.0):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian sigma.

    """

    maxy = np.real(wofz((1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)
    return (height/maxy) * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi) + const + slp*(x-cen)


def voigtfit(x,y,initpar=None,sigma=None,bounds=(-np.inf,np.inf)):
    """Fit a Voigt profile to data."""
    if initpar is None:
        initpar = [np.max(y),x[np.argmax(y)],1.0,1.0,np.median(y),0.0]
    func = voigt
    return curve_fit(func, x, y, p0=initpar, sigma=sigma, bounds=bounds)

def voigtarea(pars):
    """ Compute area of Voigt profile"""
    sig = np.maximum(pars[2],pars[3])
    x = np.linspace(-20*sig,20*sig,1000)+pars[1]
    dx = x[1]-x[0]
    v = voigt(x,np.abs(pars[0]),pars[1],pars[2],pars[3])
    varea = np.sum(v*dx)
    return varea

def ccorrelate(x,y,lag):
    """
    Cross-Correlation

    Parameters
    ----------
    x : numpy array
      First spectrum.
    y : numpy array
      Second spectrum.
    lab : numpy array
      Array of integer "lag" values at which to calculate the cross-correlation.
         For example, lag=[-3,-2,-1,0,1,2,3]

    Returns
    -------
    cc : numpy array
      The cross-correlation array.

    Example
    -------

    cc = ccorrelate(x,y,lag)

    """
    nx = len(x)
    # Subract mean values
    xd = x-np.mean(x)
    yd = y-np.mean(y)
    nlag = len(lag)
    cross = np.zeros(nlag,dtype=float)
    # Loop over lag points
    for k in range(nlag):
        # Note the reversal of the variables for negative lags.
        if lag[k]>0:
            cross[k] = np.sum(xd[0:nx-lag[k]] * yd[lag[k]:])
        else:
            cross[k] =  np.sum(yd[0:nx+lag[k]] * xd[-lag[k]:])
    cross /= np.sqrt(np.sum(xd**2)*np.sum(yd**2))
    return cross
