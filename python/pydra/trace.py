#!/usr/env python

# Imports
import numpy as np
import time
from scipy.special import erf,wofz
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks,argrelextrema,convolve2d
from scipy.ndimage import median_filter,uniform_filter
from scipy import stats
from astropy.io import fits
from scipy import ndimage
from scipy.interpolate import interp1d
from numba import njit,jit
from dlnpyutils import utils as dln,robust,mmm,coords
from matplotlib.path import Path
import matplotlib.pyplot as plt
#from . import utils,robust,mmm
from . import extract as xtract

@njit
def quadratic_coefficients(x,y):
    """ Calculate the quadratic coefficients from the three points."""
    #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    #quadratic regression statistical equation
    # y = ax**2 + b*x + c
    n = len(x)
    Sxx = np.sum(x**2) - np.sum(x)**2/n
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n
    Sxx2 = np.sum(x**3) - np.sum(x)*np.sum(x**2)/n
    Sx2y = np.sum(x**2 * y) - np.sum(x**2)*np.sum(y)/n
    Sx2x2 = np.sum(x**4) - np.sum(x**2)**2/n
    # a = ( S(x^2*y)*S(xx)-S(xy)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    # b = ( S(xy)*S(x^2x^2) - S(x^2y)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    denom = Sxx*Sx2x2 - Sxx2**2
    if denom==0:
        return [np.nan,np.nan,np.nan]
    a = ( Sx2y*Sxx - Sxy*Sxx2 ) / denom
    b = ( Sxy*Sx2x2 - Sx2y*Sxx2 ) / denom
    c = np.median(y - (a*x**2+b*x))
    coef = [a,b,c]
    return coef

@njit
def quadratic_bisector(x,y):
    """ Calculate the axis of symmetric or bisector of parabola"""
    # https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    # quadratic regression statistical equation
    n = len(x)
    if n<3:
        return None
    Sxx = np.sum(x**2) - np.sum(x)**2/n
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n
    Sxx2 = np.sum(x**3) - np.sum(x)*np.sum(x**2)/n
    Sx2y = np.sum(x**2 * y) - np.sum(x**2)*np.sum(y)/n
    Sx2x2 = np.sum(x**4) - np.sum(x**2)**2/n
    #a = ( S(x^2*y)*S(xx)-S(xy)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    #b = ( S(xy)*S(x^2x^2) - S(x^2y)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    denom = Sxx*Sx2x2 - Sxx2**2
    if denom==0:
        return np.nan
    a = ( Sx2y*Sxx - Sxy*Sxx2 ) / denom
    b = ( Sxy*Sx2x2 - Sx2y*Sxx2 ) / denom
    if a==0:
        return np.nan
    return -b/(2*a)

@njit
def unique(x):
    """
    Return the index of the unique values.
    """
    ind = np.array([0])
    if len(x)>1:
        si = np.argsort(x)
        xs = x[si]
        diff = xs[1:]-xs[:-1]
        gstep, = np.where(diff!=0)
        ind = np.concatenate((ind,gstep+1))
    return ind
    
@njit
def gvals(ycen,ysigma,y):
    """
    Calculate Gaussians in list.
    """
    nx = len(ycen)
    ny = len(y)
    model = np.zeros((ny,nx),float)
    fac = np.sqrt(2*np.pi)
    for i in range(nx):
        #  Gaussian area is A = ht*wid*sqrt(2*pi)
        # sigma = np.maximum( totflux/(ht0*np.sqrt(2*np.pi))
        amp = 1/(fac*ysigma[i])
        model[:,i] = amp*np.exp(-0.5*(y-ycen[i])**2/ysigma[i]**2)
    return model

@njit
def gradient(f):
    """
    Determine the gradient of an array.

    Parameters
    ----------
    f : numpy array
      The array to find the gradient of.

    Returns
    -------
    grad : numpy array
       The gradient of x.

    Example
    -------

    grad = gradient(f)

    """

    out = np.zeros(len(f),float)

    # Copied from numpy/lib/function_base.py gradient()
    # assuming uniform spacing
    
    # Numerical differentiation: 2nd order interior
    #slice1 = slice(1, -1)
    #slice2 = slice(None, -2)
    #slice3 = slice(1, -1)
    #slice4 = slice(2, None)
    #out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * ax_dx)
    out[1:-1] = (f[2:] - f[:-2]) / 2.0
    
    # Bottom edge
    #slice1[axis] = 0
    #slice2[axis] = 0
    #slice3[axis] = 1
    #slice4[axis] = 2
    a = -1.5
    b = 2.
    c = -0.5
    # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
    #out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    out[0] = a * f[0] + b * f[1] + c * f[2]    

    # Top edge
    #slice1[axis] = -1
    #slice2[axis] = -3
    #slice3[axis] = -2
    #slice4[axis] = -1
    a = 0.5
    b = -2.
    c = 1.5
    # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
    #out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    out[-1] = a * f[-3] + b * f[-2] + c * f[-1]    

    return out

@njit
def polyval(coef,x):
    """ Evaluation polynomial. """
    nc = len(coef)
    y = np.zeros(len(x),float)
    for i in range(nc):
        order = nc-i-1
        y += coef[i]*x**order
    return y

@njit
def gaussian(x,coef):
    """ Evaluate gaussian."""
    y = coef[0]*np.exp(-0.5*(x-coef[1])**2/coef[2]**2)
    if len(coef)==4:
        y += coef[3]
    return y

@njit
def bestscale(x,y):
    """
    Find the best scaling between x to y.
    """
    scale = np.sum(x*y)/np.sum(x**2)
    return scale

@njit
def linregression(x,y):
    """
    Linear regression
    """
    # https://en.wikipedia.org/wiki/Simple_linear_regression
    mnx = np.mean(x)
    mny = np.mean(y)
    slope = np.sum((x-mnx)*(y-mny))/np.sum((x-mnx)**2)
    yoff = mny-slope*mnx
    return slope,yoff

@njit
def gparscenpeak(y):
    """
    Estimate central position using the maximum point and
    the neighboring values.

    Parameters
    ----------
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    center : float
       Center value.

    Example
    -------
    
    cen = gparsmoments(y)

    """

    n = len(y)
    x = np.arange(n)
    maxind = np.argmax(y)
    # Now fit quadratic equation to peak and neighboring pixels
    lo = np.maximum(maxind-1,0)
    hi = np.minimum(maxind+2,n)
    cen = quadratic_bisector(x[lo:hi],y[lo:hi])
    return cen

@njit
def gparsmoments(x,y):
    """
    Estimate Gaussian center, sigma, and height using moments.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    amp : float
       Gaussian amplitude.
    center : float
       Gaussian center.
    sigma : float
       Gaussian sigma.

    Example
    -------
    
    amp,center,sigma = gparsmoments(x,y)

    """

    # If there is a large constant offset, then all the values will be off
    
    toty = np.sum(y)                            # 0th moment
    cen = np.sum(x*y)/toty                      # 1st moment
    sigma = np.sqrt(np.sum(y*(x-cen)**2)/toty)  # 2nd central moment
    # Area = ht*sigma*sqrt(2*pi)
    # ht = Area/(sigma*sqrt(2*pi))
    dx = x[1]-x[0]    
    amp = toty*dx/(sigma*np.sqrt(2*np.pi))
    return amp,cen,sigma

@njit
def gparsamppoly(usq,y):
    """
    Estimate Gaussian amplitude and offset using polynomial estimate.

    Parameters
    ----------
    usq : numpy array
       Numpy array of the square of the shifted and scaled x-values, usq=((x-cen)/sigma)^2.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    amp : float
       The Gaussian amplitude value.
    offset : float
       The constant offset.

    Example
    -------
    
    amp,offset = gparsamppoly(usq,y)

    """
    mny = np.mean(y)
    
    # These are the Gaussian polynomial coefficients for A=1 and sigma=1
    # lowest order FIRST
    f = np.array([ 9.96182626e-01, -4.74980034e-01,  9.79500447e-02, -9.77177326e-03,
                   3.78864533e-04])
    # y = offset + A*(f0 + f1*u/sigma^2 + f2*u^2/sigma^4 + f3*u^3/sigma^6 + f4*u^4/sigma^8)
    #
    # y = offset + A*(f0 + f1*(u/sigma^2) + f2*(u/sigma^2)^2 + f3*(u/sigma^2)^3 + f4*(u/sigma^2)^4)
    # us = u/sigma^2
    # y = offset + A*(f + f1*us + f2*us^2 + f3*us^3 + f4*us^4)
    # if we have an estimate for sigma, then we can solve for A and offset directly using linear OLS
    # where the term in the parentheses is the X, A is the slope and offset is the constant term
    # we can then refine to get a better sigma

    # Use initial centroid and sigma estimate to get Amplitude and offset
    fact = f[0] + f[1]*usq + f[2]*usq**2  + f[3]*usq**3  + f[4]*usq**4
    mnfact = np.mean(fact)
    amp = (np.sum(fact*y)-mnfact*mny)/(np.sum(fact**2)-mnfact**2)
    offset = amp*mnfact-mny
    
    return amp,offset

@njit
def gparssigmasearch_chisq(u,y,sigma):
    # Scale y to get Amp and offset
    yf = gaussian(u,np.array([1.0,0.0,sigma]))
    # solve for Amp and offset
    amp,off = linregression(yf,y)
    model = yf*amp+off
    residsq = (y-model)**2
    chisq = np.sum(residsq)
    return chisq

@njit
def gparssigmasearch(u,y):
    """ 
    Search a region of sigma finding the best chiqsq value.

    Parameters
    ----------
    u : numpy array
       Numpy array of the shifted x-values, u=|x-cen|.
         NOT scaled by sigma.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    amp : float
       The Gaussian amplitude value.
    sigma : float
       The Gaussian sigma value.
    offset : float
       The constant offset.
    chisq : float
       The best chisq value.

    Example
    -------
    
    amp,sigma,offset,chisq = gparssigmasearch(u,y)

    """
                  
    # Pretty fast and robust
    # 22 microseconds
    
    # Assume u extends to 3*sigma
    usq = u**2
    umax = np.max(np.abs(u))
    sigma0 = umax/3.0

    # Initialize large arrays to hold the chisq and sigma values
    sigmaarr = np.zeros(100,float)    
    chisqarr = np.zeros(100,float)
    count = 0

    # Get chisq of initial sigma value
    chisq0 = gparssigmasearch_chisq(u,y,sigma0)
    sigmaarr[count] = sigma0
    chisqarr[count] = chisq0
    count += 1
    
    # Find inner edge where chisq starts to increase
    sigma = sigma0*0.75
    last_chisq = chisq0
    flag = 0
    while (flag==0):
        chisq = gparssigmasearch_chisq(u,y,sigma)
        sigmaarr[count] = sigma
        chisqarr[count] = chisq
        count += 1
        if chisq < last_chisq:
            sigma *= 0.75
        else:
            flag = 1
        last_chisq = chisq
    low_sigma = sigma

    # Find upper edge where chisq starts to increase
    sigma = sigma0*1.25
    last_chisq = chisq0
    flag = 0
    while (flag==0):
        chisq = gparssigmasearch_chisq(u,y,sigma)
        sigmaarr[count] = sigma
        chisqarr[count] = chisq
        count += 1
        if chisq < last_chisq:
            sigma *= 1.25
        else:
            flag = 1
        last_chisq = chisq
    high_sigma = sigma

    # Trim the chisq/sigma values
    sigmaarr = sigmaarr[0:count]
    chisqarr = chisqarr[0:count]
    # sort them
    si = np.argsort(sigmaarr)
    sigmaarr = sigmaarr[si]
    chisqarr = chisqarr[si]

    bestind = np.argmin(chisqarr)
    # Fit quadratic equation to best value and neighboring points
    sigma = quadratic_bisector(sigmaarr[bestind-1:bestind+2],
                               chisqarr[bestind-1:bestind+2])

    # Now get amp and offset
    coef = np.zeros(3,float)  # numba is requiring me to do it this way
    coef[0] = 1
    coef[2] = sigma
    yf = gaussian(u,coef)
    # solve for Amp and offset
    amp,offset = linregression(yf,y)
    model = yf*amp+offset
    residsq = (y-model)**2
    chisq = np.sum(residsq)

    return amp,sigma,offset,chisq

@njit
def gparscenbisector(y,cen):
    """
    Use the symmetry of the curve to find the center

    Parameters
    ----------
    y : numpy array
       Numpy array of flux values.
    cen : float
       Initial guess of Gaussian center

    Returns
    -------
    cen: float
       Improved estimate of center

    Example
    -------
    
    cen = gparscenbisector(y,cen)

    """

    n = len(y)
    # Central pixel to start with
    xstart = int(np.ceil(cen))

    # The basic idea is that I want to find the bisector
    # of the peak
    # For each pixel on one side, I want to find the expected position at the same
    # flux level on the other side
    ind = np.arange(n)
    left = (ind < cen)
    xleft = ind[left]
    xright = ind[~left]
    yleft = y[left]
    yright = y[~left]
    # flip right values so they are ascending
    xright = np.flip(xright)
    yright = np.flip(yright)
    
    # Interpolate right flux values using left points
    #  only use values that are in the y-range of the other side    
    gdright, = np.where((yright <= np.max(yleft)) & (yright >= np.min(yleft)))
    xrightpair = np.interp(yright[gdright],yleft,xleft)

    # Interpolate left flux values using right points
    #  only use values that are in the y-range of the other side
    gdleft, = np.where((yleft <= np.max(yright)) & (yleft >= np.min(yright)))    
    xleftpair = np.interp(yleft[gdleft],yright,xright)

    # Find the mid-point for each set of flux values
    # left values and right-side pairs
    xleftmid = 0.5*(xleft[gdleft]+xleftpair)
    # right values and left-side pairs
    xrightmid = 0.5*(xright[gdright]+xrightpair)

    mids = np.concatenate((xleftmid,xrightmid))
    newcen = np.mean(mids)

    return newcen

@njit
def gparssigmaderiv(x,y,cen):
    """
    Use the first derivative to estimate sigma.

    Parameters
    ----------
    x : numpy array
       Numpy array of the x-values.
    y : numpy array
       Numpy array of flux values.
    cen : float
       The Gaussian center.

    Returns
    -------
    sigma : float
       The Gaussian sigma value.

    Example
    -------
    
    sigma = gparssigmaderiv(x,y,cen)

    """

    # can use derivative to get sigma, but need to know the offset
    # from gaussfithi5.pro
    dx = x[1]-x[0]
    dfdx = gradient(y)/dx
    u = x-cen
    r10 = dfdx/y
    slope,yoff = linregression(u,r10)
    sigma = np.sqrt(-1/slope)
    return sigma

@njit
def gparspolyiter(u,y):
    """
    Iterate back and forth between scaling in u and scaling in y.

    Parameters
    ----------
    u : numpy array
       Numpy array of the shifted x-values, u=|x-cen|.
         NOT scaled by sigma.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    amp : float
       The Gaussian amplitude value.
    sigma : float
       The Gaussian sigma value.
    offset : float
       The constant offset.

    Example
    -------
    
    amp,sigma,offset = gparspolyiter(u,y)

    """

    # NOT GOOD
    
    # These are the Gaussian polynomial coefficients for the flux (A=1 and sigma=1) if given 
    #  the sigma-scaled u squared, u^2 = ((x-cen)/sigma)^2
    #  the usq values should only to up to 9, i.e. 3 sigma
    # highest order FIRST
    ycoef = np.array([ 3.78864533e-04, -9.77177326e-03,  9.79500447e-02, -4.74980034e-01,
                       9.96182626e-01])

    # These are the Gaussian polynomial coefficients that give u-squared from scaled y (0 to 1)
    # highest orders FIRST
    ucoef = np.array([ 3.26548205e+04, -1.74691753e+05,  4.02541663e+05, -5.22876009e+05,
                       4.20862134e+05, -2.17473350e+05,  7.24096646e+04, -1.52451852e+04,
                       1.96347765e+03, -1.55782907e+02,  1.03476392e+01])

    # Assume u extends to 3*sigma
    usq = u**2
    umax = np.max(np.abs(u))
    sigma = umax/3.0
    
    # Iterate until convergence
    last_sigma = 1e10
    last_amp = 1e10
    last_off = 1e10
    count = 0
    flag = 0
    while (flag==0):
    
        # Scale y to get Amp and offset
        usclsq = (u/sigma)**2
        yf = polyval(ycoef,usclsq)
        # solve for Amp and offset
        amp,off = linregression(yf,y)
    
        # Now scale in u to get sigma
        yscl = (y-off)/amp
        # Only include "high" values, lower values are more likely to be dominated by a constant offset
        good = (yscl > 0.15)
        # This should now follow the "standard curve" and usq is just scaled by sigma^2
        # need to calculate the u-values for the "standard curve" for our y-values
        usq_standard = polyval(ucoef,yscl[good])
        # now calculate the least squares scaling factor
        #scale = bestscale(usq[good],usq_standard)
        #sigma = 1/np.sqrt(scale)
        scale,y0 = linregression(usq[good],usq_standard)
        sigma = 1/np.sqrt(scale)

        # Parameter changes
        delta_sigma = np.abs(sigma-last_sigma)/sigma*100
        delta_amp = np.abs(amp-last_amp)/amp*100
        delta_off = np.abs(off-last_off)/amp*100

        # Convergence criteria
        if (delta_sigma<1 and delta_amp<1 and delta_off<1) or count>10:
            flag = 1

        #print(count,amp,sigma,off)
                     
        # Save values for later
        last_sigma = sigma
        last_amp = amp
        last_off = off

        count += 1   # increment counter
        
    #import pdb; pdb.set_trace()

    return amp,sigma,off

#@njit
def gparspoly(u,y):
    """
    Estimate Gaussian amplitude and offset using polynomial estimate.

    Parameters
    ----------
    u : numpy array
       Numpy array of the shifted x-values, u=(x-cen).
         NOT scaled by sigma.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    amp : float
       The Gaussian amplitude value.
    sigma : float
       The Gaussian sigma value.
    offset : float
       The constant offset.

    Example
    -------
    
    amp,offset = gparspoly(u,y)

    """
    n = len(y)
    mny = np.mean(y)
    
    # These are the Gaussian polynomial coefficients for A=1 and sigma=1
    # lowest order FIRST
    f = np.array([ 9.96182626e-01, -4.74980034e-01,  9.79500447e-02, -9.77177326e-03,
                   3.78864533e-04])
    # y = offset + A*(f0 + f1*usq/sigma^2 + f2*usq^2/sigma^4 + f3*usq^3/sigma^6 + f4*usq^4/sigma^8)
    #
    # y = offset + A*(f0 + f1*(usq/sigma^2) + f2*(usq/sigma^2)^2 + f3*(usq/sigma^2)^3 + f4*(usq/sigma^2)^4)
    # us = usq/sigma^2
    # y = offset + A*(f + f1*us + f2*us^2 + f3*us^3 + f4*us^4)
    # if we have an estimate for sigma, then we can solve for A and offset directly using linear OLS
    # where the term in the parentheses is the X, A is the slope and offset is the constant term
    # we can then refine to get a better sigma

    # Fit 4th order polynomial coefficients using least-squares regression
    usq = u**2
    design = np.zeros((n,5),float)
    design[:,0] = 1
    design[:,1] = usq
    design[:,2] = usq**2
    design[:,3] = usq**3
    design[:,4] = usq**4
    # beta = (X.T * X)^-1 X.T y
    beta = np.dot(np.linalg.inv(design.T @ design),np.dot(design.T,y))

    # get multiple estimates of sigma from the higher order terms
    # beta[1] = A*f1/sigma^2
    # beta[2] = A*f2/sigma^4
    # beta[1]/beta[2]=sigma^2 * f1/f2
    #  sigma = np.sqrt(beta[1]/beta[2]*f2/f1)
    sigma12 = np.sqrt(beta[1]/beta[2]*f[2]/f[1])
    
    # Now use polynomial least-squares regression to get sigma
    # a = 1/sigma^2
    # beta[1] = A*f1*a
    # beta[2] = A*f2*a^2
    # beta[3] = A*f3*a^3
    # beta[4] = A*f4*a^4
    # divide by f and then by beta[1]
    # fbeta = beta[1:]/f[1:]
    # fbeta[0] = A*a
    # fbeta[1] = A*a^2
    # fbeta[2] = A*a^3
    # fbeta[3] = A*a^4
    # and then divide by beta2[0] to get rid of Amp
    # fabeta = fbeta[1:]/fbeta[0]
    # fabeta[1] = a
    # fabeta[2] = a^2
    # fabeta[3] = a^3    
    fbeta = beta[1:]/f[1:]
    fabeta = fbeta[1:]/fbeta[0]

    # a = 1/sigma^2
    # y = offset + A*(f0 + f1*a*usq + f2*a^2*usq^2 + f3*a^3*usq^3 + f4*a^4*usq^4)
    ampsigprod = beta[1]/f[1]  # A*a

    # I CAN'T FIND A SOLUTION FOR THIS
    
    # Use initial centroid and sigma estimate to get Amplitude and offset
    fact = f[0] + f[1]*usq + f[2]*usq**2  + f[3]*usq**3  + f[4]*usq**4
    mnfact = np.mean(fact)
    amp = (np.sum(fact*y)-mnfact*mny)/(np.sum(fact**2)-mnfact**2)
    offset = amp*mnfact-mny
    
    return amp,offset

@njit
def gparslog(x,y):
    """
    Estimate Gaussian parameters by taking natural log of the Y-values.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    amp : float
       Gaussian amplitude.
    center : float
       Gaussian center.
    sigma : float
       Gaussian sigma.

    Example
    -------
    
    amp,center,sigma = gparslog(x,y)

    """

    # This assumes there is NO constant offset
    # If there is, then the amplitude and sigma will be off.
    
    # Directly solve for the parameters using ln(y)
    #  y = A*exp(-0.5*(x-x0)^2/sigma^2)
    #  ln(y) = ln(A)-0.5*(x-x0)^2/sigma^2
    #        = -0.5/sigma^2 * x^2 + x0/sigma^2 * x + ln(A)-0.5*x0^2/sigma^2
    #  fit quadratic equation in x
    #  a = -0.5/sigma^2              quadratic term
    #  b = x0/sigma^2                linear term
    #  c = ln(A)-0.5*x0^2/sigma^2    constant term
    #  ->   sigma=sqrt(-1/(2*a))
    #  ->   x0=b*sigma**2
    #  ->  A=exp(c + x0**2/(2*sigma**2))
    lny = np.log(y)
    quadcoef = quadratic_coefficients(x,lny)
    sigma = np.sqrt(-1/(2*quadcoef[0]))
    cen = quadcoef[1]*sigma**2
    amp = np.exp(quadcoef[2]+cen**2/(2*sigma**2))
    return amp,cen,sigma

@njit
def gparssigmahalfflux(u,y):
    """
    Estimate Gaussian sigma by finding the x-position where the flux has reached
    half of the maximum.

    Parameters
    ----------
    u : numpy array
       Numpy array of scaled x-values, u=|x-cen|.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    sigma : float
       Gaussian sigma.

    Example
    -------
    
    sigma = gparssigmahalfflux(u,y)


    """
    # This assumes there is no constant offset
    n = len(u)
    si = np.argsort(u)
    uu = u[si]
    yy = y[si]
    maxy = np.max(yy)
    halfmaxy = 0.5*maxy
    lowind = np.where(yy < halfmaxy)[0]
    if len(lowind)>0:
        lo = lowind[0]        
    else:
        lo = n-1
    hiind = np.where(yy >= halfmaxy)[0]    
    hi = hiind[-1]
    # linearly interpolate between the two points
    slp = (yy[hi]-yy[lo])/(uu[hi]-uu[lo])
    yoff = yy[hi]-slp*uu[hi]
    xhalf = (halfmaxy-yoff)/slp
    fwhm = 2*xhalf
    sigma = fwhm/2.35
    return sigma
    
@njit
def gparssigmaarea(x,y,amp):
    """
    Estimate Gaussian sigma using the area and estimate of amplitude.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
       Numpy array of flux values.
    amp : float
       Estimate of Gaussian amplitude.

    Returns
    -------
    sigma : float
       Gaussian sigma.

    Example
    -------
    
    sigma = gparssigmaarea(x,y,amp)

    """
    # This assumes that there is no constant offset
    # and that we have the full Gaussian out to +/-3 sigma
    dx = x[1]-x[0]
    totflux = np.sum(y)
    sigma = totflux*dx/(amp*np.sqrt(2*np.pi))
    return sigma

@njit
def gparssigmascale(usq,yscl):
    """
    Estimate Gaussian sigma by scaling u/y.

    Parameters
    ----------
    usq : numpy array
       Numpy array of the square of the shifted x-values, u=(x-cen).
         This u is NOT scaled by sigma.
    yscl : numpy array
       Numpy array of scaled Y-values, yscl=(y-offset)/amp

    Returns
    -------
    amp : float
       Gaussian amplitude.
    center : float
       Gaussian center.
    sigma : float
       Gaussian sigma.

    Example
    -------
    
    amp,center,sigma = gparssigmascale(usq,yscl)

    """

    # Get improved sigma estimate
    # we are going to "invert" the problem and scale u to get the sigma
    # highest orders FIRST
    ycoef = np.array([ 3.26548205e+04, -1.74691753e+05,  4.02541663e+05, -5.22876009e+05,
                       4.20862134e+05, -2.17473350e+05,  7.24096646e+04, -1.52451852e+04,
                       1.96347765e+03, -1.55782907e+02,  1.03476392e+01])
    # Only include "high" values, lower values are more likely to be dominated by a constant offset
    good = (yscl > 0.15)
    # This should now follow the "standard curve" and u is just scaled by sigma^2
    # need to calculate the u-values for the "standard curve" for our y-values
    ustandard = polyval(ycoef,yscl[good])
    # now calculate the least squares scaling factor
    scale = np.sum(usq[good]*ustandard)/np.sum(ustandard**2)
    sigma = np.sqrt(scale)
    return sigma

@njit
def gparsampdiffs(u,y):
    """
    Estimate Gaussian amplitude and offset using the known flux differences
    of pixels near the peak.

    Parameters
    ----------
    u : numpy array
       Numpy array of the shifted and sigma-scaled x-values, u=|(x-cen)/sigma|.
    y : numpy array
       Numpy array of the Y-values.

    Returns
    -------
    amp : float
       Gaussian amplitude.
    offset : float
       Constant offset

    Example
    -------
    
    amp,offset = gparsampdiffs(u,y)

    """

    # This method is quite robust, as long as you have a decent sigma estimate
    
    # Use flux differences (then the offset drops out) between pixels near the peak
    #  to estimate the amplitude and constant offset
    udiff = u.reshape(-1,1) - u.reshape(1,-1)             # all u differences
    ydiff = y.reshape(-1,1) - y.reshape(1,-1)             # all y differences
    ygauss = gaussian(u,np.array([1.0,0.0,1.0]))          # Gaussian flux values
    ygdiff = ygauss.reshape(-1,1) - ygauss.reshape(1,-1)  # Gaussian flux differences
    # Now find the best scale of the two to get the Gaussian amplitude
    scale = np.sum(ydiff*ygdiff)/np.sum(ydiff**2)
    amp = 1/scale
    # Now estimate the constant offset
    offset = np.mean(y-amp*ygauss)
    return amp,offset

@njit
def gaussxccenter(x,y,coef):
    """
    Perform cross-correlation with Gaussian to estimate improved center.
    A maximum lag of +/-3 pixels is used.  This is then refined with
    a second cross-correlation with a maximum lag of +/-0.3 pixels in 0.1
    pixel steps.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
       Numpy array of flux values.
    coef : numpy array
       Numpy array of Gaussian parameters, [amp,center,sigma].

    Returns
    -------
    center : float
       Cross-correlation center.

    Example
    -------
    
    center = gaussxccenter(x,y,coef)

    """
    maxlag = 3
    # It's better to subtract off a constant term
    #  than to just include it in the model
    if len(coef)==4:
        yp = y-coef[3]
    else:
        yp = y
    dx = x[1]-x[0]
    nlag1 = maxlag*2+1
    nx = len(x)
    lag1 = np.arange(nlag1)-maxlag
    # Extend x by +/-maxlog on the two sides
    xext = np.concatenate((-np.arange(maxlag,0,-1)*dx+x[0],x))
    xext = np.append(xext,np.arange(1,maxlag+1)*dx+x[-1])
    model = coef[0]*np.exp(-0.5*(xext-coef[1])**2/coef[2]**2)
    ccf1 = np.zeros(nlag1,float)
    for i in range(nlag1):
        ccf1[i] = np.sum(yp*model[maxlag-lag1[i]:maxlag+nx-lag1[i]])
    shift1 = quadratic_bisector(lag1,ccf1)
    cen1 = coef[1]+shift1
    
    # Now refine, in 0.1 dx steps
    maxlag2 = 3
    nlag2 = maxlag2*2+1
    lag2 = np.arange(nlag2)-maxlag2
    ccf2 = np.zeros(nlag2,float)
    for i in range(nlag2):
        xoff = cen1+0.1*lag2[i]*dx
        model2 = coef[0]*np.exp(-0.5*(x-xoff)**2/coef[2]**2)
        ccf2[i] = np.sum(yp*model2)            
    shift2 = quadratic_bisector(lag2,ccf2)
    cen2 = cen1 + shift2*0.1*dx
    
    return cen2

@njit
def gparsmodelscale(x,y,coef):
    """
    Estimate Gaussian amplitude and offset by scaling a model.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
       Numpy array of flux values.
    coef : numpy array
       Numpy array of Gaussian parameters [height,center,sigma].

    Returns
    -------
    amp : float
       The Gaussian amplitude value.
    offset : float
       The constant offset.

    Example
    -------
    
    amp,offset = gparsmodelscale(x,y,coef)

    """
    n = len(x)
    model = gaussian(x,coef)
    mnx = np.mean(model)
    sumx2 = np.sum((model-mnx)**2)
    mny = np.mean(y)
    sumxy = np.sum((model-mnx)*(y-mny))
    slope = sumxy/sumx2
    yoff = mny-slope*mnx
    return slope,yoff
    
@njit
def gpars1(x,y):
    """
    Simple Gaussian fit using 8th order polynomial estimate.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    pars : numpy array
       Gaussian parameters [height, center, sigma].  If multiple
       profiles are input, then the output dimensions are [Nprofiles,3].

    Example
    -------
    
    pars = gpars1(x,y)

    """

    # This takes about 85 microseconds for 10 points
    
    yp = np.maximum(y,0)
    good, = np.where(np.isfinite(y))
    xp = x[good]
    yp = yp[good]
    ymax = np.max(yp)
    ymin = np.min(yp)
    xmax = xp[np.argmax(yp)]

    # Use moments to get first estimates, then iterate
    # subtract minimum in case there is an offset
    # fast: 2 microseconds
    amp0,cen0,sigma0 = gparsmoments(xp,yp)
    good = ((np.abs(x-cen0) <= 3*sigma0) & np.isfinite(y))
    # Re-estimate center and sigma with reduced range
    #  only if the range changed a lot
    if len(yp)/np.sum(good) > 1.5:
        amp0,cen0,sigma0 = gparsmoments(x[good],yp[good])
    # Our Gaussian might be truncated on one side
    nxsigpos = ((np.max(x[good])-cen0)/sigma0)
    nxsigneg = ((cen0-np.min(x[good]))/sigma0)
    if nxsigpos<0 or nxsigneg<0 or nxsigpos<2 or nxsigneg<2 or np.abs(nxsigpos-nxsigneg)>0.5:
        # Reflect points about the maximum
        x2 = np.concatenate((xp-xmax,xmax-xp))+xmax
        y2 = np.concatenate((yp,yp))
        # Need to sort or otherwise the dx in gparsmoments() will be wrong
        si = np.argsort(x2)
        x2,y2 = x2[si],y2[si]
        # Need to get unique x-values, otherwise the amplitude will be artificially
        #  inflated because we added more points
        #_,ui = np.unique(x2,return_index=True)
        ui = unique(x2) 
        x2,y2 = x2[ui],y2[ui]
        amp,cen0,sigma0 = gparsmoments(x2,y2)
        good = ((np.abs(x-cen0) <= 3*sigma0) & np.isfinite(y))
        
    xp = x[good]
    yp = y[good]
    u0 = (xp-cen0)**2
    
    # Use sigma search
    u0 = np.abs(xp-cen0)
    amp1,sigma1,offset1,chisq1 = gparssigmasearch(u0,yp)

    # Improve center estimate with cross-correlation
    # somewhat slower: 24 microseconds
    coef1 = np.zeros(4,float)
    coef1[0] = amp1
    coef1[1] = cen0
    coef1[2] = sigma1
    coef1[3] = offset1
    cen1 = gaussxccenter(xp,yp,coef1)
    
    # Iterate: Use sigma search
    u1 = np.abs(xp-cen1)
    amp2,sigma2,offset2,chisq2 = gparssigmasearch(u1,yp)   

    # Make sure new solution is better
    coef = np.zeros(4,float)
    if chisq2 <= chisq1:
        coef[0] = amp2
        coef[1] = cen1
        coef[2] = sigma2
        coef[3] = offset2
    # New solution is worse, revert to prevous one
    else:
        coef[0] = amp1
        coef[1] = cen0
        coef[2] = sigma1
        coef[3] = offset1
    
    # Get initial estimate of amplitude and offset from flux differences
    # fast, 12 micro seconds
    #A0,offset0 = gparsampdiffs(np.abs(xp-cen0)/sig0,yp)
    
    # Use Gaussian polynomial approximation to estimate amplitude and offset
    #A0,offset0 = gparspoly(u0/sig0**2,yp)
    
    # Get improved sigma estimate using u/y scaling
    # fast: 12 microseconds
    #sig1 = gparssigmascale(u0,(yp-offset0)/A0)
    
    # Iterate: Re-estimate center with cross-correlation
    # somewhat slower: 24 microseconds
    #coef0 = np.array([A0,cen0,sig1,offset0])
    #cen1 = gaussxccenter(xp,yp,coef0)
    #u1 = (xp-cen1)**2
    
    # Iterate: Re-estimate A and offset with the improved sigma value
    ##A1,offset1 = gparspoly(u1/sig1**2,yp)
    #A1,offset1 = gparsampdiffs(np.abs(xp-cen1)/sig1,yp)
    # doesn't improve really
    
    # fit amplitude and offset by using a linear fit of the model
    # (slope is the amplitude and y-offset is the offset term)
    # fast: 2 microseconds
    #tcoef = np.array([1.0,cen1,sig1])
    #A2,offset2 = gparsmodelscale(xp,yp,tcoef)
    # doesn't improve really

    return coef
    
@njit
def gpars1log(x,y):
    """
    Simple Gaussian fit to central 5 pixel values.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
       Numpy array of flux values.

    Returns
    -------
    pars : numpy array
       Gaussian parameters [height, center, sigma].  If multiple
       profiles are input, then the output dimensions are [3].

    Example
    -------
    
    pars = gpars1(x,y)

    """
    pars = np.zeros(3,float)
    nx = len(y)
    nhalf = nx//2
    gd = (np.isfinite(y) & (y>0))
    #if np.sum(gd)<5:
    x = x[gd]
    y = y[gd]
    totflux = np.sum(y)
    ht0 = y[nhalf]
    if np.isfinite(ht0)==False:
        ht0 = np.max(y)
    # Use flux-weighted moment to get center
    cen1 = np.sum(y*x)/totflux
    #  Gaussian area is A = ht*wid*sqrt(2*pi)
    sigma1 = np.maximum( totflux/(ht0*np.sqrt(2*np.pi)) , 0.01)
    # Use linear-least squares to calculate height and sigma
    psf = np.exp(-0.5*(x-cen1)**2/sigma1**2)          # normalized Gaussian
    wtht = np.sum(y*psf)/np.sum(psf*psf)          # linear least squares
    
    # Directly solve for the parameters using ln(y)
    pars = gparslog(x,y)

    return pars

@njit
def gpars(xprofiles,yprofiles,npixprofiles):
    """
    Simple Gaussian fit to central pixel values.

    Parameters
    ----------
    xprofiles : numpy array
       Numpy array of x-values.
    yprofiles : numpy array
       Numpy array of flux values.  Can be a single or
       mulitple profiles.  If mulitple profiles, the dimensions
       should be [Nprofiles,5].
    
    
    Returns
    -------
    pars : numpy array
       Gaussian parameters [height, center, sigma].  If multiple
       profiles are input, then the output dimensions are [Nprofiles,3].

    Example
    -------
    
    pars = gpars(x,y)

    """
    
    #if y.ndim==1:
    #    nprofiles = 1
    #    y = np.atleast_1d(y)
    #else:
    #    nprofiles = y.shape[0]
    #nx = y.shape[1]
    #nhalf = nx//2
    nprofiles = len(npixprofiles)
    # Loop over profiles
    pars = np.zeros((nprofiles,4),float)
    for i in range(nprofiles):
        # First try the central 5 pixels first        
        #x1 = x[i,2:7]
        #y1 = y[i,2:7]
        #x1 = x[i,:]
        #y1 = y[i,:]
        #x1 = profiles[i]['x']
        #y1 = profiles[i]['y']
        npix = npixprofiles[i]
        x1 = xprofiles[i,:npix]
        y1 = yprofiles[i,:npix]
        #gd = (np.isfinite(y1) & (y1>0))
        #if np.sum(gd)<5:
        #    x1 = x1[gd]
        #    y1 = y1[gd]
        pars1 = gpars1(x1,y1)
        # If sigma is too high, then expand to include more points     
        #if pars1[2]>2:
        #    x1 = x[i,:]
        #    y1 = y[i,:]
        #    gd = (np.isfinite(y1) & (y1>0))
        #    if np.sum(gd)<9:
        #        x1 = x1[gd]
        #        y1 = y1[gd]
        #    pars1 = gpars1(x1,y1)
        # Put results in big array
        pars[i,:] = pars1
        
    return pars

@njit
def traceboundary(y,cen):
    """
    Find the boundary of the trace.
    Can be where the values start to increase again, or
    hit the edge of the array
    """

    n = len(y)
    peak = y[cen]
    
    # Walk to lower boundary
    flag = 0
    count = 0
    last_y = peak
    last_position = cen
    position = cen-1
    while (flag==0):
        y1 = y[position]        
        # need to stop
        if y1<0.5*peak and y1>=last_y:
            flag = 1
            low = position+1
        # at lower edge, stop
        elif position==0:
            flag = 1
            low = position
        # keep going
        else:
            position -= 1
        count += 1   # increment counter
        # keep last values
        last_y = y1
        last_position = position
        
    # Walk to upper boundary
    flag = 0
    count = 0
    last_y = peak
    last_position = cen
    position = cen+1
    while (flag==0):
        y1 = y[position]
        # need to stop
        if y1<0.5*peak and y1>=last_y:
            flag = 1
            high = position-1
        elif position==n-1:
            flag = 1
            high = position
        # keep going
        else:
            position += 1
        count += 1   # increment counter
        # keep last values
        last_y = y1
        last_position = position
    
    return low,high

        
def traceim(im,nbin=50,minsigheight=3,minheight=None,hratio2=0.97,
            neifluxratio=0.3,verbose=False):
    """
    Find traces in the image.  The dispersion dimension is assumed
    to be along the X-axis.

    Parameters
    ----------
    im : numpy array
       2-D image to find the traces in.
    nbin : int, optional
       Number of columns to bin to find peaks.  Default is 50.
    minsigheight : float, optional
       Height threshold for the trace peaks based on standard deviation
       in the background pixels (median filtered).  Default is 3.
    minheight : float, optional
       Absolute height threshold for the trace peaks.  Default is to
       use `minsigheight`.
    hratio2 : float, optional
       Ratio of peak height to the height two pixels away in the
       spatial profile, e.g. height_2pixelsaway < 0.8*height_peak.
       Default is 0.8.
    neifluxratio : float, optional
       Ratio of peak flux in one column block and the next column
       block.  Default is 0.3.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    xmed : numpy array
       X-values for MEDIM.
    medim : numpy array
       The median image of the columns.
    smedim : numpy array
       Slightly smoothed version of medim along Y-axis.
    peaks : numpy array
       The 2-D peaks image.
    pmask : numpy array
       The 2-D boolean mask image for the trace peak pixels.
    xpind : numpy array
       X-values for peak mask pixels.
    ypind : numpy array
       Y-values for peak mask pixels.       
    xindex : index
       The X-values index of peaks.

    Example
    -------

    xmed,medim,peaks,pmask,xpind,ypind,xindex = traceim(im)

    """
    im = im.astype(float)  # make sure it is float64, needed for njit to work right sometimes
    # This assumes that the dispersion direction is along the X-axis
    #  Y-axis is spatial dimension
    ny,nx = im.shape
    y,x = np.arange(ny),np.arange(nx)
    # Median filter in nbin column blocks all in one shot
    medim1 = dln.rebin(im,binsize=[1,nbin],med=True)
    smedim1 = uniform_filter(medim1,[3,1])  # average slightly in Y-direction
    xmed1 = dln.rebin(x,binsize=nbin,med=True)
    # half-steps
    medim2 = dln.rebin(im[:,nbin//2:],binsize=[1,nbin],med=True)
    smedim2 = uniform_filter(medim2,[3,1])  # average slightly in Y-direction
    xmed2 = dln.rebin(x[nbin//2:],binsize=nbin,med=True)
    # Splice them together
    medim = dln.splice(medim1,medim2,axis=1)
    smedim = dln.splice(smedim1,smedim2,axis=1)    
    xmed = dln.splice(xmed1,xmed2,axis=0)
    nxb = medim.shape[1]

    # Compare flux values to neighboring spatial pixels
    #  and two pixels away
    # Shift and mask wrapped values with NaN
    rollyp2 = dln.roll(smedim,2,axis=0)
    rollyp1 = dln.roll(smedim,1,axis=0)
    rollyn1 = dln.roll(smedim,-1,axis=0)
    rollyn2 = dln.roll(smedim,-2,axis=0)
    rollxp1 = dln.roll(smedim,1,axis=1)
    rollxn1 = dln.roll(smedim,-1,axis=1)
    if minheight is not None:
        height_thresh = minheight
    else:
        height_thresh = 0.0
    peaks = ( (((smedim >= rollyn1) & (smedim > rollyp1)) |
               ((smedim > rollyn1) & (smedim >= rollyp1))) &
              (rollyp2 < hratio2*smedim) & (rollyn2 < hratio2*smedim) &
	      (smedim > height_thresh))
    
    # Now require trace heights to be within ~30% of at least one neighboring
    #   median-filtered column block
    peaksht = ((np.abs(smedim[peaks]-rollxp1[peaks])/smedim[peaks] < neifluxratio) |
              (np.abs(smedim[peaks]-rollxn1[peaks])/smedim[peaks] < neifluxratio))
    ind = np.where(peaks.ravel())[0][peaksht]
    ypind,xpind = np.unravel_index(ind,peaks.shape)
    xindex = dln.create_index(xpind)
    
    # Boolean mask image for the trace peak pixels
    pmask = np.zeros(smedim.shape,bool)
    pmask[ypind,xpind] = True
    
    # Compute height threshold from the background pixels
    #  need the traces already to do this
    if minheight is None:
        # Grow peak mask by 7 pixels in spatial and 5 in dispersion direction
        #  the rest of the pixels are background
        exclude_mask = convolve2d(pmask,np.ones((7,5)),mode='same')
        backmask = (exclude_mask < 0.5)
        backpix = smedim[backmask]
        medback,sigback = backvals(backpix)
        if np.isfinite(medback)==False:
            medback = np.nanmedian(backpix)
        if np.isfinite(sigback)==False:
            sigback = np.nanstd(backpix)
        # Use the final background values to set the height threshold
        height_thresh = medback + minsigheight * sigback
        if verbose: print('height threshold',height_thresh)
        # Impose this on the peak mask
        #oldpmask = pmask
        #pmask = (smedim*pmask > height_thresh)
        #ypind,xpind = np.where(pmask)
        oldpmask = pmask
        ypind,xpind = np.where((smedim*pmask > height_thresh))
        # Create the X-values index of peaks
        xindex = dln.create_index(xpind)

    return xmed,medim,smedim,peaks,pmask,xpind,ypind,xindex

def tracematch(xmed,medim,smedim,peaks,pmask,xpind,ypind,xindex,
               neifluxratio=0.3,verbose=False):
    """
    Matches peaks of traces across column blocks.

    Parameters
    ----------
    xmed : numpy array
       X-values for MEDIM.
    medim : numpy array
       The median image of the columns.
    smedim : numpy array
       Slightly smoothed version of medim along Y-axis.
    peaks : numpy array
       The 2-D peaks image.
    pmask : numpy array
       The 2-D boolean mask image for the trace peak pixels.
    xpind : numpy array
       X-values for peak mask pixels.
    ypind : numpy array
       Y-values for peak mask pixels. 
    xindex : index
       The X-values index of peaks.

    Returns
    -------
    tracelist : list
       List of the trace information

    Example
    -------

    tracelist = tracematch()

    """

    ny,nxb = medim.shape
    nbin = int(2*(xmed[1]-xmed[0]))
    
    # Now match up the traces across column blocks
    # Loop over the column blocks and either connect a peak
    # to an existing trace or start a new trace
    # only connect traces that haven't been "lost"
    tracelist = []
    # Looping over unique column blocks, not every one might be represented
    for i in range(len(xindex['value'])):
        ind = xindex['index'][xindex['lo'][i]:xindex['hi'][i]+1]
        nind = len(ind)
        xind = xpind[ind]
        yind = ypind[ind]
        yind.sort()    # sort y-values
        xb = xind[0]   # this column block index        

        # Deal with neighbors
        #  we shouldn't have any neighbors in Y
        if len(ind)>1:
            diff = dln.slope(np.array(yind))
            bd, = np.where(diff == 1)
            if len(bd)>0:
                torem = []
                for j in range(len(bd)):
                    lo = bd[j]
                    hi = lo+1
                    if medim[yind[lo],xb] >= medim[yind[hi],xb]:
                        torem.append(hi)
                    else:
                        torem.append(lo)
                yind = np.delete(yind,np.array(torem))
                xind = np.delete(xind,np.array(torem))            
                nind = len(yind)

        if verbose: print(i,xb,len(xind),'peaks')
        # Adding traces
        #   Loop through all of the existing traces and try to match them
        #   to new ones in this row
        tymatches = []
        for j in range(len(tracelist)):
            itrace = tracelist[j]
            y1 = itrace['yvalues'][-1]
            x1 = itrace['xbvalues'][-1]
            deltax = xb-x1
            # Only connect traces that haven't been lost
            if deltax<3:
                # Check the pmask boolean image to see if it connects
                ymatch = None
                if pmask[y1,xb]:
                    ymatch = y1
                elif pmask[y1-1,xb]:
                    ymatch = y1-1                       
                elif pmask[y1+1,xb]:
                    ymatch = y1+1

                if ymatch is not None:
                    itrace['yvalues'].append(ymatch)
                    itrace['xbvalues'].append(xb)
                    itrace['xvalues'].append(xmed[xb])
                    itrace['heights'].append(smedim[ymatch,xb])
                    itrace['ncol'] += 1                    
                    tymatches.append(ymatch)
                    if verbose: print(' Trace '+str(j)+' Y='+str(ymatch)+' matched')
                else:
                    # Lost
                    if itrace['lost']:
                        itrace['lost'] = True
                        if verbose: print(' Trace '+str(j)+' LOST')
            # Lost, separation too large
            else:
                # Lost
                if itrace['lost']:
                    itrace['lost'] = True
                    if verbose: print(' Trace '+str(j)+' LOST')        
        # Add new traces
        # Remove the ones that matched
        yleft = yind.copy()
        if len(tymatches)>0:
            ind1,ind2 = dln.match(yind,tymatches)
            nmatch = len(ind1)
            if nmatch>0 and nmatch<len(yind):
                yleft = np.delete(yleft,ind1)
            else:
                yleft = []
        # Add the ones that are left
        if len(yleft)>0:
            if verbose: print(' Adding '+str(len(yleft))+' new traces')
        for j in range(len(yleft)):
            # Skip traces too low or high
            if yleft[j]<2 or yleft[j]>(ny-3):
                continue
            itrace = {'index':0,'ncol':0,'yvalues':[],'xbvalues':[],'xvalues':[],'heights':[],
                      'lost':False}
            itrace['index'] = len(tracelist)+1
            itrace['ncol'] = 1
            itrace['nbin'] = nbin  # save the binning information
            itrace['yvalues'].append(yleft[j])
            itrace['xbvalues'].append(xb)
            itrace['xvalues'].append(xmed[xb])                
            itrace['heights'].append(smedim[yleft[j],xb])
            tracelist.append(itrace)
            if verbose: print(' Adding Y='+str(yleft[j]))
            
    # For each trace, check if it continues at the end, but below the nominal height threshold
    for i in range(len(tracelist)):
        itrace = tracelist[i]
        # Check lower X-values
        flag = 0
        nnew = 0
        x1 = itrace['xbvalues'][0]
        y1 = itrace['yvalues'][0]
        if x1==0 or x1==(nxb-1): continue  # at edge already
        while (flag==0):
            # Check peaks, heights must be within 30% of the height of the last one
            doesconnect = peaks[y1-1:y1+2,x1-1] & (np.abs(smedim[y1-1:y1+2,x1-1]-smedim[y1,x1])/smedim[y1,x1] < neifluxratio)
            if np.sum(doesconnect)>0:
                newx = x1-1
                if doesconnect[1]==True:
                    newy = y1
                elif doesconnect[0]==True:
                    newy = y1-1
                else:
                    newy = y1+1
                # Add new column to the trace (to beginning)
                itrace['yvalues'].insert(0,newy)
                itrace['xbvalues'].insert(0,newx)
                itrace['xvalues'].insert(0,xmed[newx])
                itrace['heights'].insert(0,smedim[newy,newx])
                itrace['ncol'] += 1
                if verbose:
                    print(' Trace '+str(i)+' Adding X='+str(newx)+' Y='+str(newy))
                # Update x1/y1
                x1 = newx
                y1 = newy
                # If at edge, stop                
                if newx==0 or newx==(nxb-1): flag = 1
                nnew += 1
            # No match
            else:
                flag = 1
        # Stuff it back in
        tracelist[i] = itrace
        
        # Check higher X-values
        flag = 0
        nnew = 0
        x1 = itrace['xbvalues'][-1]
        y1 = itrace['yvalues'][-1]
        if x1==0 or x1==(nxb-1): continue  # at edge already
        while (flag==0):
            # Check peaks, heights must be within 30% of the height of the last one
            doesconnect = peaks[y1-1:y1+2,x1+1] & (np.abs(smedim[y1-1:y1+2,x1+1]-smedim[y1,x1])/smedim[y1,x1] < neifluxratio)
            if np.sum(doesconnect)>0:
                newx = x1+1
                if doesconnect[1]==True:
                    newy = y1
                elif doesconnect[0]==True:
                    newy = y1-1
                else:
                    newy = y1+1
                # Add new column to the trace
                itrace['yvalues'].append(newy)
                itrace['xbvalues'].append(newx)
                itrace['xvalues'].append(xmed[newx])
                itrace['heights'].append(smedim[newy,newx])
                itrace['ncol'] += 1
                if verbose:
                    print(' Trace '+str(i)+' Adding X='+str(newx)+' Y='+str(newy))
                # Update x1/y1
                x1 = newx
                y1 = newy
                # If at edge, stop                
                if newx==0 or newx==(nxb-1): flag = 1
                nnew += 1
            # No match
            else:
                flag = 1      
        # Stuff it back in
        tracelist[i] = itrace

    return tracelist

def tracegauss(medim,tracelist):
    """
    Fit spectral trace and PSF using a Gaussian.
    The input image should only include one spectrum/trace.

    Parameters
    ----------
    medim : numpy array
       The median image of the columns.
    tracelist : list
       List of the trace information.

    Returns
    -------
    tracelist : list
       List of the trace information.

    Example
    -------

    tracelist = tracegauss(im,tracelist):

    """

    ny,nxb = medim.shape
    
    # Calculate Gaussian parameters
    nprofiles = np.sum([tr['ncol'] for tr in tracelist])
    yprofiles = np.zeros((nprofiles,50),float) + np.nan  # all bad to start
    xprofiles = np.zeros((nprofiles,50),int)
    npixprofiles = np.zeros(nprofiles,int)
    profiles = []
    trindex = np.zeros(nprofiles,int)
    count = 0
    for t,tr in enumerate(tracelist):
        for i in range(tr['ncol']):
            xlo,xhi = traceboundary(medim[:,tr['xbvalues'][i]],tr['yvalues'][i])
            #if tr['yvalues'][i]<4:        # at bottom edge
            #    lo = 4-tr['yvalues'][i]
            #    yp = np.zeros(9,float)+np.nan
            #    yp[lo:] = medim[:tr['yvalues'][i]+5,tr['xbvalues'][i]]                
            #elif tr['yvalues'][i]>(ny-5):   # at top edge
            #    hi = 9-(ny-tr['yvalues'][i])+1
            #    yp = np.zeros(9,float)+np.nan
            #    yp[:hi] = medim[tr['yvalues'][i]-4:,tr['xbvalues'][i]]
            #else:
            #    lo = tr['yvalues'][i]-4
            #    hi = tr['yvalues'][i]+5
            #    yp = medim[lo:hi,tr['xbvalues'][i]]
            #profiles[count,:] = yp
            #xp = np.arange(9)+tr['yvalues'][i]-4            
            #xprofiles[count,:] = xp
            trindex[count] = t
            xp = np.arange(xlo,xhi+1)
            yp = medim[xlo:xhi+1,tr['xbvalues'][i]]
            npix = len(xp)
            xprofiles[i,:npix] = xp
            yprofiles[i,:npix] = yp
            npixprofiles[i] = npix
            #prof1 = {'x':np.arange(xlo,xhi+1),
            #         'y':medim[xlo:xhi+1,tr['xbvalues'][i]],'index':t}
            #profiles.append(prof1)
            count += 1

    # Trim the arrays
    maxnpix = np.max(npixprofiles)
    xprofiles = xprofiles[:,:maxnpix]
    yprofiles = yprofiles[:,:maxnpix]

    # Get Gaussian parameters for all profiles at once
    #pars = gpars(xprofiles,profiles)
    pars = gpars(xprofiles,yprofiles,npixprofiles)
    # Stuff the Gaussian parameters into the list
    count = 0
    for tr in tracelist:
        tr['gyheight'] = np.zeros(tr['ncol'],float)
        tr['gycenter'] = np.zeros(tr['ncol'],float)
        tr['gysigma'] = np.zeros(tr['ncol'],float)
        tr['gyoffset'] = np.zeros(tr['ncol'],float)        
        for i in range(tr['ncol']):
            tr['gyheight'][i] = pars[count,0]
            tr['gycenter'][i] = pars[count,1]
            tr['gysigma'][i] = pars[count,2]
            tr['gyoffset'][i] = pars[count,3]            
            count += 1
            
    # Fit trace coefficients
    for tr in tracelist:
        tr['tcoef'] = None
        tr['xmin'] = np.min(tr['xvalues'])-tr['nbin']//2
        tr['xmax'] = np.max(tr['xvalues'])+tr['nbin']//2
        xt = np.array(tr['xvalues'])
        # Trace Y-position
        yt = tr['gycenter']
        norder = 3
        coef = np.polyfit(xt,yt,norder)
        resid = yt-np.polyval(coef,xt)
        std = np.std(resid)
        # Check if we need to go to 4th order
        if std>0.1:
            norder = 4
            coef = np.polyfit(xt,yt,norder)
            resid = yt-np.polyval(coef,xt)
            std = np.std(resid)
        # Check if we need to go to 5th order
        if std>0.1:
            norder = 5
            coef = np.polyfit(xt,yt,norder)
            resid = yt-np.polyval(coef,xt)
            std = np.std(resid)
        tr['tcoef'] = coef
        tr['tstd'] = std
        # Fit Gaussian sigma as well
        syt = tr['gysigma']
        norder = 2
        coef = np.polyfit(xt,syt,norder)
        resid = syt-np.polyval(coef,xt)
        std = np.std(resid)
        # Check if we need to go to 3rd order
        if std>0.05:
            norder = 3
            coef = np.polyfit(xt,syt,norder)
            resid = syt-np.polyval(coef,xt)
            std = np.std(resid)
        # Check if we need to go to 4th order
        if std>0.05:
            norder = 4
            coef = np.polyfit(xt,syt,norder)
            resid = syt-np.polyval(coef,xt)
            std = np.std(resid)
        tr['sigcoef'] = coef
        tr['sigstd'] = std
        
    return tracelist
    
def hornetrace(im,x=None,y=None,ytrace=None,err=None,off=10,backoff=50,smlen=31):
    """
    Fit spectral trace and PSF using the Horne 1986 method.
    The input image should only include one spectrum/trace.

    Parameters
    ----------
    im : numpy array
       The image with the spectrum to trace.
    x : numpy array, optional
       The X-array for IM.
    y : numpy array, optional
       The Y-array for IM.

    Returns
    -------
    psf : numpy array
       The 2D PSF image.

    Example
    -------

    psf = hornetrace(im,x,y):

    """

    ny,nx = im.shape
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)

    # Find the trace
    if ytrace is None:
        xmed,medim,smedim,peaks,pmask,xpind,ypind,xindex = traceim(im)        
        tracelist = tracematch(xmed,medim,smedim,peaks,pmask,xpind,ypind,xindex)
        if len(tracelist)>1:
            medheights = [np.median(t['heights']) for t in tracelist]
            bestind = np.argmax(medheights)
            tr = tracelist[bestind]
        else:
            tr = tracelist[0]
        tcoef = np.polyfit(tr['xvalues'],tr['yvalues'],3)
        ytrace = np.polyval(tcoef,np.arange(nx))
        # Get Gaussian parameters for all profiles at once
        pars = gpars(xprofiles,profiles)

    ## Figure out how many pixel to use in the profile
    #maxsigma = np.max(tr['gysigma'])
    #if nprofile is None:
    #    # want 2.0*sigma on each side
    #    nprofile = int(np.ceil(4*maxsigma))
    #    if nprofile % 2 != 0:
    #        nprofile += 1
    #else:
    #    # Make sure nprofile is even
    #    if nprofile % 2 != 0:
    #        nprofile += 1
    #nhalf = nprofile//2
        
    yest = np.nanmedian(ytrace)
    # Get the subimage
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    # Background subtract
    med = np.nanmedian(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    subim = subim.astype(float)   # make sure they are float64    
    if err is not None:
        suberr = err[yblo:ybhi,:]
        suberr = suberr.astype(float)            
    # Mask other parts of the image
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    sim = subim*mask
    if err is not None:
        serr = suberr*mask
        badpix = (serr <= 0)
        serr[badpix] = 1e20
    # Compute the profile/probability matrix from the image
    tot = np.nansum(np.maximum(sim,0),axis=0)
    tot[(tot<=0) | ~np.isfinite(tot)] = 1
    psf1 = np.maximum(sim,0)/tot
    psf = np.zeros(psf1.shape,float)
    for i in range(nback):
        psf[i,:] = dln.medfilt(psf1[i,:],smlen)
        #psf[i,:] = utils.gsmooth(psf1[i,:],smlen)        
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    totpsf = np.nansum(psf,axis=0)
    totpsf[(totpsf<=0) | (~np.isfinite(totpsf))] = 1
    psf /= totpsf
    psf[(psf<0) | ~np.isfinite(psf)] = 0

    return psf
    
def marshtrace(im,x=None,y=None,nprofile=None,kind='medfilt',wings='gaussian'):
    """
    Fit spectral trace and PSF using the Marsh 1989 method.
    The input image should only include one spectrum/trace.

    Parameters
    ----------
    im : numpy array
       The image with the spectrum to trace.
    x : numpy array, optional
       The X-array for IM.
    y : numpy array, optional
       The Y-array for IM.
    nprofile : int, optional
       Y-profile size.  Default is None and it is determined by the Gaussian sigma.
    kind : str, optional
      Kind of description to use of the PSF: "poly" or "medfilt".  Default is "medfilt".
    wings : str, optional
      The type of wings to use: "gaussian", or "none".  Default is "gaussian".

    Returns
    -------
    model : numpy array
       The 2D PSF image.
    recmodel : numpy array
       The 2D rectified PSF image.
    coefarr : numpy array
       The polynomial coefficients.  Only if kind is "poly".

    Example
    -------

    model,recmodel = marshtrace(im,x,y):

    """

    # TO DO:
    # -use input x/y arrays
    # -background subtraction
    # -gaussian wings
    
    ny,nx = im.shape
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)

    # Find the trace
    xmed,medim,smedim,peaks,pmask,xpind,ypind,xindex = traceim(im)        
    tracelist = tracematch(xmed,medim,smedim,peaks,pmask,xpind,ypind,xindex)
    if len(tracelist)>1:
        medheights = [np.median(t['heights']) for t in tracelist]
        bestind = np.argmax(medheights)
        tr = [tracelist[bestind]]
    tracelist = tracegauss(medim,tracelist)
    tr = tracelist[0]
    ytrace = np.polyval(tr['tcoef'],np.arange(nx))

    # Figure out how many pixel to use in the profile
    maxsigma = np.max(tr['gysigma'])
    if nprofile is None:
        # want 2.0*sigma on each side
        nprofile = int(np.ceil(4*maxsigma))
        if nprofile % 2 == 0:
            nprofile += 1
    else:
        # Make sure nprofile is even
        if nprofile % 2 == 0:
            nprofile += 1
    nhalf = nprofile//2
    
    
    # -resample each column onto the integer rows
    # -can fit polynomial to each row, just like Horne/Marshall method
    # -to make the model, get the values on the integer rows, and then
    #  resample onto the correct shifted values

    #yest = np.nanmedian(ytrace)
    ## Get the subimage
    #yblo = int(np.maximum(yest-backoff,0))
    #ybhi = int(np.minimum(yest+backoff,ny))
    #nback = ybhi-yblo
    ## Background subtract
    #med = np.nanmedian(im[yblo:ybhi,:],axis=0)
    #medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    #subim = im[yblo:ybhi,:]-medim
    #subim = subim.astype(float)   # make sure they are float64    
    #off = 10
    ## Mask other parts of the image
    #ylo = ytrace-off - yblo
    #yhi = ytrace+off - yblo
    #yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    #mask = (yy >= ylo) & (yy <= yhi)
    #sim = subim*mask

    # Get the rectified profiles
    profiles = np.zeros((tr['ncol'],nprofile),float)+np.nan
    xprofiles = np.zeros((tr['ncol'],nprofile),float)
    dxprofiles = np.zeros((tr['ncol'],nprofile),float)
    xprofile = np.arange(-nhalf,nhalf+1)
    yprofile = np.zeros((tr['ncol'],nprofile),float)
    for i in range(tr['ncol']):
        if tr['yvalues'][i]<nhalf:        # at bottom edge
            ylo = 0
            yhi = tr['yvalues'][i]+nhalf+1
            lo = nprofile-(yhi-ylo)
            hi = nprofile
            yp = np.zeros(nprofile,float)+np.nan
            yp[lo:hi] = medim[ylo:yhi,tr['xbvalues'][i]]                
        elif tr['yvalues'][i]>(ny-nhalf-1):   # at top edge
            ylo = tr['yvalues'][i]-nhalf
            yhi = ny
            lo = 0
            hi = yhi-ylo
            yp = np.zeros(nprofile,float)+np.nan
            yp[lo:hi] = medim[ylo:yhi,tr['xbvalues'][i]]
        else:
            ylo = tr['yvalues'][i]-nhalf
            yhi = tr['yvalues'][i]+nhalf+1
            yp = medim[ylo:yhi,tr['xbvalues'][i]]
        profiles[i,:] = yp
        xp = np.arange(nprofile)+tr['yvalues'][i]-nhalf
        xprofiles[i,:] = xp            
        dxp = xp-np.polyval(tr['tcoef'],tr['xvalues'][i])
        dxprofiles[i,:] = dxp
        good = np.isfinite(yp)
        yout = dln.interp(dxp[good],yp[good],xprofile,kind='quadratic',extrapolate=False)
        yprofile[i,:] = yout  #/np.nansum(yout)

    ngood = np.sum(np.isfinite(yprofile),axis=0)
    goodrows, = np.where(ngood > 0.9*profiles.shape[0])
    tot = np.sum(yprofile[:,goodrows],axis=1)
    yprofile /= tot.reshape(-1,1)   # normalizing

    from dlnpyutils import plotting as pl    
    import pdb; pdb.set_trace()
    
    # Fit each rectified row
    #   with polynomial or median filter
    coefarr = np.zeros((nprofile,6),float)
    recmodel = np.zeros((nprofile,nx),float)
    xx = np.arange(nx)
    for i in range(nprofile):
        good = np.isfinite(yprofile[:,i])
        if np.sum(good)>0.5*tr['ncol']:
            xp = np.array(tr['xvalues'])[good]
            yp = yprofile[good,i]
            # Median filter
            if kind.lower()=='medfilt':
                medpsf1 = dln.medfilt(yp,7)
                # need to extrapolate to the ends
                yout = dln.interp(xp,medpsf1,xx,kind='quadratic',extrapolate=True)
                recmodel[i,:] = yout
            # Polynomial fitting
            elif kind.lower()=='poly' or kind.lower()=='polynomial':
                medy = median_filter(yp,7)
                # calculate average sigma so chisq~Npoints
                # also, the rms
                sig = np.sqrt(np.sum((yp-medy)**2)/len(yp))
                rms = np.zeros(5,float)
                chisq = np.zeros(5,float)
                coef = np.zeros((5,6),float)
                bic = np.zeros(5,float)
                pv = np.zeros(5,float)
                for j in range(5):
                    coef[j,-(j+2):] = np.polyfit(xp,yp,j+1)
                    mp = np.polyval(coef[j,:],xp)
                    rms[j] = np.sqrt(np.mean((yp-mp)**2))
                    chisq[j] = np.sum((yp-mp)**2/sig**2)
                    lnl = -0.5*np.sum((yp-mp)**2/sig**2)-np.pi*len(yp)
                    bic[j] = (j+2)*np.log(len(yp))-2*lnl
                    pv[j] = stats.distributions.chi2.sf(chisq[j],i+2)
                bestind = np.argmin(bic)
                coefarr[i,:] = coef[bestind,:]
                recmodel[i,:] = np.polyval(coef[bestind,:],xx)

                # Use Bayesian Information Criterion (BIC)
                #  to decide which order to use
                # BIC = k*ln(n) - 2*ln(L)
                # k = number of model parameters
                # n = number of data points
                # L = the maximized value of the likelihood function of the model
                #   = -0.5*chisq - 0.5*Sum(2*pi*sigma_i**2)
                
            else:
                raise ValueError(str(kind)+' not supported')

    from dlnpyutils import plotting as pl
    import pdb; pdb.set_trace()
            
    # Normalize
    recmodel /= np.sum(recmodel,axis=0).reshape(1,-1)
    
    # Now make the full model
    model = np.zeros((ny,nx),float)
    for i in range(nx):
        ymid = np.polyval(tr['tcoef'],xx[i])
        if ymid<nhalf:        # at bottom edge
            ylo = 0
            yhi = int(np.round(ymid)+nhalf+1)
            lo = nprofile-(yhi-ylo+1)
            hi = nprofile
        elif ymid>(ny-nhalf-1):   # at top edge
            ylo = int(nhalf-np.round(ymid))
            yhi = ny
            lo = 0
            hi = yhi-ylo+1
        else:
            ylo = int(np.round(ymid)-nhalf)  # index into full 2D array
            yhi = int(np.round(ymid)+nhalf+1)
            lo = 0                       # index for 9-hight recmodel array
            hi = nprofile

        xp = np.arange(ylo,yhi)
        dxp = xp-ymid
        yout = dln.interp(xprofile[lo:hi],recmodel[lo:hi,i],dxp,kind='quadratic',
                          extrapolate=False,fill_value=0.0)
        model[ylo:yhi,i] = yout

    if kind.lower()=='medfilt':
        return model,recmodel
    elif kind.lower()=='poly' or kind.lower()=='polynomial':
        return model,recmodel,coefarr      

        
    
class Traces():
    """
    Class to hold multiple spectral traces.
    """

    def __init__(self,data=None):
        # Input a trace list or an image
        self._data = None
        self.index = None
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None        
        # Data input
        if data is not None:
            # List of dictionaries or Trace objects
            if type(data) is list:
                self._data = data
            # Image input, find the traces
            elif type(data) is np.ndarray and data.ndim==2:
                tlist = trace(data)
                self._data = tlist
            self.index = np.arange(len(self._data))+1
            # Create the trace objects
            if isinstance(self._data[0],Trace)==False:
                data = self._data
                del self._data
                self._data = []
                for i in range(len(data)):
                    t = Trace(data[i])
                    t.index = i+1
                    self._data.append(t)

    def model(self,x=None,yr=None):
        """ Make 2D model of the normalized PSFs."""
        if x is None:
            xr = [0,int(np.ceil(self.xmax))]
            x = np.arange(xr[1]-xr[0]+1)+xr[0]
        if yr is None:
            yr = [0,int(np.ceil(self.ymax))+10]
        y = np.arange(yr[1]-yr[0]+1)+yr[0]
        nx = len(x)
        ny = len(y)
        psf = np.zeros((ny,nx),float)
        for i,t in enumerate(self):
            sigma1 = np.median(t.sigma)
            yr1 = [int(np.floor(t.ymin-3*sigma1)),
                   int(np.ceil(t.ymax+3*sigma1))]
            xr1 = [int(np.floor(t.xmin)),
                   int(np.ceil(t.xmax))]
            x1 = np.arange(xr1[1]-xr1[0]+1)+xr1[0]
            psf1 = t.model(x1,yr=yr1)
            slc1 = slice(yr1[0]-yr[0], yr1[1]-yr[0]+1)
            slc2 = slice(xr1[0]-xr[0], xr1[1]-xr[0]+1)
            psf[(slc1,slc2)] += psf1
        return psf
        
    def __len__(self):
        if self._data is None:
            return 0
        else:
            return len(self._data)

    @property
    def hasdata(self):
        if self._data is None:
            return False
        else:
            return True
        
    @property
    def xmin(self):
        if self._xmin is None:
            xmin = np.inf
            for t in self:
                xmin = min(xmin,t.xmin)
            self._xmin = xmin
        return self._xmin

    @property
    def xmax(self):
        if self._xmax is None:
            xmax = -np.inf
            for t in self:
                xmax = max(xmax,t.xmax)
            self._xmax = xmax
        return self._xmax

    @property
    def ymin(self):
        if self._ymin is None:
            ymin = np.inf
            for t in self:
                ymin = min(ymin,t.ymin)
            self._ymin = ymin
        return self._ymin

    @property
    def ymax(self):
        if self._ymax is None:
            ymax = -np.inf
            for t in self:
                ymax = max(ymax,t.ymax)
            self._ymax = ymax
        return self._ymax    
        
    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        if self.hasdata:
            body = '{:d} traces, '.format(len(self))
            body += 'X=[{:.1f},{:.1f}]'.format(self.xmin,self.xmax)
            body += ',Y=[{:.1f},{:.1f}]'.format(self.ymin,self.ymax)
        else:
            body = ''
        out = ''.join([prefix, body, ')']) +'\n'
        return out
        
    def __getitem__(self,item):
        # Single trace
        if type(item) is int:
            if item > len(self)-1:
                raise IndexError('Traces index out of range')
            # Already trace object
            if isinstance(self._data[item],Trace):
                tr = self._data[item]
            # Dictionary, make Trace
            else:
                tr = Trace(self._data[item])
                tr.index = item+1
            return tr
        # Multiple traces
        elif type(item) is tuple or type(item) is slice:
            index = np.arange(len(self))
            index = index[item]  # apply it to the indices
            data = len(index)*[None]
            for i in range(len(index)):
                data[i] = self._data[index[i]]  # by reference
            tr = Traces(data)
            tr.index = index+1
            return tr
        else:
            raise ValueError('index not understood')

    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < len(self):
            self._count += 1            
            return self[self._count-1]
        else:
            raise StopIteration

    def footprint(self,xr=None,yr=None):
        """ Return an image that gives the footprints of the traces """
        # The value will be the trace number
        if xr is None:
            xr = [0,int(np.ceil(self.xmax))]
        if yr is None:
            yr = [0,int(np.ceil(self.ymax))+10]
        nx = xr[1]-xr[0]+1
        ny = yr[1]-yr[0]+1
        foot = np.zeros((ny,nx),int)
        for i,t in enumerate(self):
            slc = t.slice(nsigma=3)
            mk = t.mask(nsigma=3)
            foot[slc][mk] = t.index
        return foot

    def overlap(self,nsigma=1):
        """ Check overlap of traces within a certain sigma level."""
        xr = [0,int(np.ceil(self.xmax))]
        yr = [0,int(np.ceil(self.ymax))+10]
        nx = xr[1]-xr[0]+1
        ny = yr[1]-yr[0]+1
        foot = np.zeros((ny,nx),int)
        olap = []
        for i in range(len(self)):
            tr = self[i]
            slc = tr.slice(nsigma=nsigma)
            msk = tr.mask(nsigma=nsigma)
            if np.sum(foot[slc][msk])>0:
                ol = ((foot[slc]>0) & msk)
                tindex = np.unique(foot[slc][ol])
                if len(tindex)==1: tindex=tindex[0]
                olap.append((i+1,tindex))
            foot[slc][msk] = tr.index
        return olap
    
    def neighbors(self,xr=None,yr=None):
        """ Find overlap of neighbors within 5 sigma."""
        if xr is None:
            xr = [0,int(np.ceil(self.xmax))]
        if yr is None:
            yr = [0,int(np.ceil(self.ymax))+10]
        nx = xr[1]-xr[0]+1
        ny = yr[1]-yr[0]+1
        foot = np.zeros((ny,nx),int)
        nolap = np.zeros((ny,nx),int)
        olap = np.zeros((ny,nx),bool)
        trace1im = np.zeros((ny,nx),int)
        trace2im = np.zeros((ny,nx),int)        
        #olap = np.zeros((ny,nx,2),int)        
        for i,t in enumerate(self):
            slc = t.slice(nsigma=5)
            msk = t.mask(nsigma=5)
            # Overlap
            #   [slc] returns a 2D image
            #   [mk] returns a 1D subset of that
            if np.sum(foot[slc][msk])>0:
                ol = ((foot[slc]>0) & msk)
                olap[slc][ol] = True
                nolap[slc][ol] += 1
                trace1im[slc][ol] = np.unique(foot[slc][ol])[0]  # previous value
                trace2im[slc][ol] = t.index
                foot[slc][msk] = t.index
            # No overlap, add this trace index
            else:
                foot[slc][mk] = t.index
        olapindex = np.stack((trace1im,trace2im))
        return foot,olap,nolap,olapindex
    
    def extract(self,im,recenter=False):
        """ Extract the spectrum from an image."""
        pass
        
    def copy(self):
        return Traces(self._data.copy())
        
    def plot(self,**kwargs):
        for t in self:
            t.plot(**kwargs)
        
class Trace():
    """
    Class for a single spectral trace.
    """
    
    def __init__(self,data=None):
        self._data = data
        self._polygondict = {}
        self._maskdict = {}

    def __call__(self,x=None,xr=None,step=1):
        """ Return the trace path."""
        # [N,2] x/y values
        if self._data is None:
            raise ValueError('No trace data yet')
        if x is None and xr is None:
            x = np.arange(np.floor(self._data['xmin']),np.ceil(self._data['xmax'])+1,step)
        if x is None and xr is not None:
            x = np.arange(xr[0],xr[1])
        # Get the curve/path
        y = np.polyval(self._data['tcoef'],x)
        out = np.zeros((len(x),2))
        out[:,0] = x
        out[:,1] = y
        return out     

    def model(self,xr=None,yr=None):
        """ Return normalized 2D image model."""
        if xr is None:
            xr = [int(np.ceil(self.xmin)),int(np.floor(self.xmax))]
        if yr is None:
            sigma = np.median(self.sigma)
            yr = [int(np.floor(np.min(self.y)-3*sigma)),
                  int(np.ceil(np.max(self.y)+3*sigma))]
        if xr[0]!=xr[1]:
            x = np.arange(xr[0],xr[1])  # do not include end point
        else:
            x = np.array([xr[0]])
        y = np.arange(yr[0],yr[1])
        ycen = np.polyval(self._data['tcoef'],x)
        ysigma = np.polyval(self._data['sigcoef'],x)
        # subtract minimum y
        ycenrel = ycen - yr[0]
        yrel = y-yr[0]
        psf = gvals(ycenrel,ysigma,yrel)
        return psf
            
    def __len__(self):
        if self._data is None:
            return 0
        else:
            return 1

    @property
    def size(self):
        if self._data is None:
            return 0
        else:
            return len(self._data)
        
    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        if self.hasdata:
            body = 'X=[{:.1f},{:.1f}]'.format(self.xmin,self.xmax)
            body += ',Y=[{:.1f},{:.1f}]'.format(self.ymin,self.ymax)
        else:
            body = ''
        out = ''.join([prefix, body, ')']) +'\n'
        return out

    def __array__(self):
        """ Return the main data array."""
        return self.data

    @property
    def hasdata(self):
        if self._data is None:
            return False
        else:
            return True

    @property
    def x(self):
        if self._data is None:
            return None
        else:
            return self._data['xvalues']

    @property
    def y(self):
        if self._data is None:
            return None
        else:
            return self._data['yvalues']        
        
    @property
    def data(self):
        if self.hasdata==False:
            return None
        return self()[:,1]

    @property
    def pars(self):
        if self.hasdata==False:
            return None
        return self._data['tcoef']
    
    @property
    def xmin(self):
        if self.hasdata==False:
            return None
        return self._data['xmin']

    @property
    def xmax(self):
        if self.hasdata==False:
            return None
        return self._data['xmax']    

    @property
    def ymin(self):
        if self.hasdata==False:
            return None
        return np.min(self()[:,1])

    @property
    def ymax(self):
        if self.hasdata==False:
            return None
        return np.max(self()[:,1])        
    
    #@property
    #def sigma(self):
    #    if self.hasdata==False:
    #        return None
    #    return self._data['gysigma']

    @property
    def coef(self):
        if self.hasdata==False:
            return None
        else:
            return self._data['tcoef']

    @property
    def sigcoef(self):
        if self.hasdata==False:
            return None
        else:
            return self._data['sigcoef']        

    def sigma(self,x=None,xr=None):
        """ Return Gaussian sigma for x values."""
        if x is None and xr is None:
            x = np.arange(self.xmin,self.xmax)
        if x is None and xr is not None:
            x = nplarange(xr[0],xr[1])
        return np.polyval(self.sigcoef,x)

    def amplitude(self,x=None,xr=None):
        """ Return Gaussian amplitude for x values."""
        if x is None and xr is None:
            x = np.arange(self.xmin,self.xmax)
        if x is None and xr is not None:
            x = nplarange(xr[0],xr[1])
        sig = self.sigma(x)
        # Gaussian area is A = ht*wid*sqrt(2*pi)
        # ht = 1/(sigma*np.sqrt(2*np.pi)
        return 1/(sig*np.sqrt(2*np.pi))
    
    def fit(self,image,initpos=None):
        """ Fit a trace to data."""
        if initpos is not None:
            if len(initpos)!=2:
                raise ValueError("Initial guess position must have 2 elements (X,Y)")
        # Find tracing in the image
        trlist = trace(image)
        # Find the traces closest to the initial guess position
        medheight = np.zeros(len(trlist),float)
        dist = np.zeros(len(trlist),float)+999999.
        for i,tr in enumerate(trlist):
            medheight[i] = np.nanmedian(tr['heights'])
            if initpos is not None:
                if tr['xmin'] <= initpos[0] and tr['xmax'] >= initpos[0]:
                    yval = np.polyval(tr['tcoef'],initpos[0])
                    dist[i] = np.abs(yval-initpos[1])
        # Using initial position
        if initpos is not None:
            bestind = np.argmin(dist)
        else:
            # use brightest trace, based on height
            bestind = np.argmax(medheight)
        self._data = trlist[bestind]
    
    def dispersionaxis(self):
        """ Determine the dispersion axis."""
        pass
        # Determine dispersion axis
        #  the axis with no ambiguities/degeneracies, single-valued

    def polygon(self,nsigma=2.5,step=50):
        """ Return the polygon of the trace area."""
        # cache result to make future calls faster
        key = '{:.1f}-{:d}'.format(nsigma,step)
        pts = self._polygondict.get(key)
        if pts is None:
            pts = self.dopolygon(nsigma=nsigma,step=step)
            self._polygondict[key] = pts
        return pts
        
    def dopolygon(self,nsigma=2.5,step=50):
        """ Return the polygon of the trace area."""        
        cpts = self(step=step)  # center of fiber, x/y
        sigma = np.polyval(self._data['sigcoef'],cpts[:,0])
        x = np.append(cpts[:,0],np.flip(cpts[:,0]))
        y = np.append(cpts[:,1]-nsigma*sigma,np.flip(cpts[:,1]+nsigma*sigma))
        pts = np.stack((x,y)).T  # [N,2]
        return pts

    def slice(self,nsigma=2.5):
        """
        Return a slice object that can be used to get the part of an
        image needed for this trace.
        """
        pts = self.polygon(nsigma=nsigma)
        ymin = int(np.floor(np.min(pts[:,1])))
        ymax = int(np.ceil(np.max(pts[:,1])))
        xmin = int(np.floor(np.min(pts[:,0])))
        xmax = int(np.ceil(np.max(pts[:,0])))        
        return slice(ymin,ymax),slice(xmin,xmax)

    def mask(self,nsigma=2.5):
        """ Return a mask to use to mask an image."""
        # cache result to make future calls faster
        key = '{:.1f}'.format(nsigma)
        msk = self._maskdict.get(key)
        if msk is None:
            msk = self.domask(nsigma=nsigma)
            self._maskdict[key] = msk
        return msk
        
    def domask(self,nsigma=2.5):
        """ Return a mask to use to mask an image."""
        # This should be used with the image return in combination with slice()
        slcy,slcx = self.slice(nsigma=nsigma)
        pts = self.polygon(nsigma=nsigma,step=50)
        nx = slcx.stop-slcx.start
        ny = slcy.stop-slcy.start
        x = np.arange(0,nx)
        y = np.arange(0,ny)
        xpoly = pts[:,0]-slcx.start
        ypoly = pts[:,1]-slcy.start
        xx,yy = np.meshgrid(x,y)
        # find the points that are inside
        tupVerts = list(zip(xpoly,ypoly))
        points = np.vstack((xx.ravel(),yy.ravel())).T
        p = Path(tupVerts) # make a polygon
        inside = p.contains_points(points)  # this takes some time, 17ms
        msk = inside.reshape(ny,nx)
        return msk

    def overlap(self,tr,nsigma=1.0):
        """ Check if this trace overlaps another one."""
        if isinstance(tr,Trace)==False:
            raise ValueError("Input is not a Trace object")
        pts1 = self.polygon(nsigma=nsigma)
        pts2 = tr.polygon(nsigma=nsigma)        
        return coords.doPolygonsOverlap(pts1[:,0],pts1[:,1],pts2[:,0],pts2[:,1])

    def optimal(self,im):
        """ Measure the optimal PSF on the image."""
        pts = self()
        ytrace = pts[:,1]
        psf = xtract.optimalpsf(im,ytrace,off=10,backoff=50,smlen=31)
        return psf
    
    def extract(self,image,errim=None,kind='psf',recenter=False,resize=False,skyfit=False):
        """
        Extract the spectrum from an image.

        Parameters
        ----------
        image : numpy array
           The 2D image with the spectrum to extract.
        errim : numpy array, optional
           The 2D uncertainty image for "image".
        kind : str, optional
          The type of extraction to perform:
            boxcar, psf, or optimal
        recenter : bool, optional
           Recenter the PSF on the image.  Default is False.
        resize : bool, optional
           Resize the Gaussian sigma value of the trace.  Default is False.
        skyfit : bool, optional
           Fit background for each column.  Default is False.

        Returns
        -------
        tab : table
           Table of extracted results.  At minimum has "flux".  The
             exact columns will depend on the input parameters.

        Example
        -------
       
        tab = extract(im,err,kind='psf')

        """
        # Get subimage
        slc = self.slice(nsigma=5)
        msk = self.mask(nsigma=5)
        xr = [slc[1].start,slc[1].stop]
        yr = [slc[0].start,slc[0].stop]
        im = image[slc].copy()
        im *= msk  # apply the mask
        if errim is not None:
            err = errim[slc]
        else:
            err = None
        psf = self.model(xr=xr,yr=yr)
        # Recenter/Resize the trace
        if recenter or resize:
            sh = im.shape
            xhalf = sh[1]//2
            # First fit the PSF peak
            medpsf = np.median(psf[:,xhalf-50:xhalf+50],axis=1)
            psfgpars,psfgcov = dln.gaussfit(np.arange(len(medpsf)),medpsf,binned=True)
            # Then fit the image peak
            medim = np.median(im[:,xhalf-50:xhalf+50],axis=1)
            initpar = psfgpars
            pkind = np.argmax(medpsf)
            initpar[0] = medim[pkind]
            bounds = [np.zeros(4,float)-np.inf,np.zeros(4,float)+np.inf]
            bounds[0][0] = 0
            bounds[0][1] = initpar[1]-2
            bounds[1][1] = initpar[1]+2            
            bounds[0][2] = 0
            imgpars,imgcov = dln.gaussfit(np.arange(len(medim)),medim,initpar=initpar,binned=True,bounds=bounds)
            # Make new shifted trace object
            tr = self.copy()
            if recenter:
                shift = imgpars[1]-psfgpars[1]
                tr._data['tcoef'][-1] += shift
            if resize:
                scale_sigma = imgpars[2]/psfgpars[2]                
                tr._data['sigcoef'] *= scale_sigma
            # Get new slice/mask/subimage, etc.
            slc = tr.slice(nsigma=5)
            msk = tr.mask(nsigma=5)
            xr = [slc[1].start,slc[1].stop]
            yr = [slc[0].start,slc[0].stop]
            im = image[slc].copy()
            im *= msk  # apply the mask
            if errim is not None:
                err = errim[slc]
            else:
                err = None
            psf = tr.model(xr=xr,yr=yr)
        # Use original trace
        else:
            tr = self
            
        # Do the extraction
        # -- Boxcar --
        if kind=='boxcar':
            pmsk = (msk & (psf > 0.005))
            boxflux = np.nansum(pmsk*im,axis=0)
            if err is not None:
                boxerr = np.sqrt(np.nansum(msk*err**2,axis=0))
                dt = [('flux',float),('err',float)]
                tab = np.zeros(len(boxflux),dtype=np.dtype(dt))
                tab['flux'] = boxflux
                tab['err'] = boxerr
            else:
                dt = [('flux',float)]
                tab = np.zeros(len(boxflux),dtype=np.dtype(dt))
                tab['flux'] = boxflux
                
        # -- Optimal extraction --
        elif kind=='optimal':
            pts = tr(xr=xr)
            ytrace = pts[:,1]  # absolute 
            ytrace -= yr[0]    # for subimage
            pmsk = (msk & (psf > 0.005))
            im *= pmsk
            out = xtract.extract_optimal(im,ytrace,imerr=err,verbose=False)
            flux,fluxerr,trace,opsf = out
            dt = [('flux',float),('err',float),('trace',float),('sky',float),('skyerr',float)]
            tab = np.zeros(len(flux),dtype=np.dtype(dt))
            tab['flux'] = flux
            tab['err'] = fluxerr
            tab['trace'] = trace
            #tab['sky'] = sky
            #tab['skyerr'] = skyerr
            
        # -- PSF extraction --
        elif kind=='psf':
            import pdb; pdb.set_trace()
            out = xtract.extract_psf(im,psf,err=err,skyfit=skyfit)
            if skyfit:
                flux,fluxerr,sky,skyerr = out
                dt = [('flux',float),('err',float),('sky',float),('skyerr',float)]
                tab = np.zeros(len(flux),dtype=np.dtype(dt))
                tab['flux'] = flux
                tab['err'] = fluxerr
                tab['sky'] = sky
                tab['skyerr'] = skyerr
            else:
                flux,fluxerr = out
                dt = [('flux',float),('err',float)]
                tab = np.zeros(len(flux),dtype=np.dtype(dt))
                tab['flux'] = flux
                tab['err'] = fluxerr                
        # -- Spectro-perfectionism --
        elif kind=='perfectionism':
            pts = tr(xr=xr)
            ytrace = pts[:,1]  # absolute 
            ytrace -= yr[0]    # for subimage
            pmsk = (msk & (psf > 0.005))
            im *= pmsk
            out = xtract.spectroperfectionism(im,ytrace,imerr=None,verbose=False)
            flux,fluxerr,trace,psf = out
            dt = [('flux',float),('err',float)]
            tab = np.zeros(len(flux),dtype=np.dtype(dt))
            tab['flux'] = flux
            tab['err'] = fluxerr            
            
        return tab
                                      
    def copy(self):
        return Trace(self._data.copy())

    def read(self,filename):
        """ Read from a file."""
        pass

    def save(self,filename):
        """ Write to a file."""
        pass
    
    def plot(self,**kwargs):
        pts = self()
        plt.plot(pts[:,0],pts[:,1],**kwargs)


def backvals(backpix):
    """ Estimate the median and sigma of background pixels."""
    vals = backpix.copy()
    medback = np.nanmedian(vals)
    sigback = dln.mad(vals)
    lastmedback = 1e30
    lastsigback = 1e30
    done = False
    # Outlier rejection
    count = 0
    while (done==False):
        gdback, = np.where(vals < (medback+3*sigback))
        vals = vals[gdback]
        medback = np.nanmedian(vals)
        sigback = dln.mad(vals)
        if sigback <= 0.0:
            sigback = np.std(vals)
        if np.abs(medback-lastmedback) < 0.01*lastmedback and \
           np.abs(sigback-lastsigback) < 0.01*lastsigback:
            done = True
        if count>10: done = True
        lastmedback = medback
        lastsigback = sigback
        count += 1
    return medback,sigback

def trace(im,nbin=50,minsigheight=3,minheight=None,hratio2=0.97,
          neifluxratio=0.3,mincol=0.5,verbose=False):
    """
    Find traces in the image.  The dispersion dimension is assumed
    to be along the X-axis.

    Parameters
    ----------
    im : numpy array
       2-D image to find the traces in.
    nbin : int, optional
       Number of columns to bin to find peaks.  Default is 50.
    minsigheight : float, optional
       Height threshold for the trace peaks based on standard deviation
       in the background pixels (median filtered).  Default is 3.
    minheight : float, optional
       Absolute height threshold for the trace peaks.  Default is to
       use `minsigheight`.
    hratio2 : float, optional
       Ratio of peak height to the height two pixels away in the
       spatial profile, e.g. height_2pixelsaway < 0.8*height_peak.
       Default is 0.8.
    neifluxratio : float, optional
       Ratio of peak flux in one column block and the next column
       block.  Default is 0.3.
    mincol : float or int, optional
       Minimum number of column blocks required to keep a trace.
       Note that float values <= 1.0 are considered fractions of
       total number of columns blocks. Default is 0.5.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    tracelist : list
       List of traces and their information.

    Example
    -------

    tracelist = trace(im)

    """
    
    if minsigheight < 0:
        raise ValueError('minsigheight must be positive')
    if minheight is not None and minheight < 0:
        raise ValueError('minheight must be positive')    
    if hratio2 < 0.1 or hratio2 > 0.99:
        raise ValueError('hratio2 must be >0.1 and <0.9')
    if neifluxratio < 0 or neifluxratio > 0.9:
        raise ValueError('neifluxratio must be >0.0 and <0.9')

    # How to set the height threshold so it will work properly in the two
    # limiting cases?
    # 1) Few traces, most pixels are dominated by background
    # 2) Lots of traces, most pixels are dominated by the traces.
    # Also, the trace flux can be weaker at the edges.
    # We use our regular criteria to find good peaks (without any
    # threshold).  This already limits the pixels quite a bit.
    # Then we grow these to exclude pixels near the traces.
    # Those pixels not "excludes" are assumed to be "background"
    # pixels and used to determine the median and standard deviation
    # of the background.  The height threshold is then determined
    # from the these two values, i.e.
    # height_thresh = medback + nsig * sigback
    
    # This assumes that the dispersion direction is along the X-axis
    #  Y-axis is spatial dimension
    ny,nx = im.shape
    y,x = np.arange(ny),np.arange(nx)
    # Median filter in nbin column blocks all in one shot
    medim1 = dln.rebin(im,binsize=[1,nbin],med=True)
    smedim1 = uniform_filter(medim1,[3,1])  # average slightly in Y-direction
    xmed1 = dln.rebin(x,binsize=nbin,med=True)
    # half-steps
    medim2 = dln.rebin(im[:,nbin//2:],binsize=[1,nbin],med=True)
    smedim2 = uniform_filter(medim2,[3,1])  # average slightly in Y-direction
    xmed2 = dln.rebin(x[nbin//2:],binsize=nbin,med=True)
    # Splice them together
    medim = dln.splice(medim1,medim2,axis=1)
    smedim = dln.splice(smedim1,smedim2,axis=1)    
    xmed = dln.splice(xmed1,xmed2,axis=0)
    nxb = medim.shape[1]
    # the smedim is smoothly slightly in Y and will change the profile
    # use medim to fit the Gaussians
    
    # Compare flux values to neighboring spatial pixels
    #  and two pixels away
    # Shift and mask wrapped values with NaN
    rollyp2 = dln.roll(smedim,2,axis=0)
    rollyp1 = dln.roll(smedim,1,axis=0)
    rollyn1 = dln.roll(smedim,-1,axis=0)
    rollyn2 = dln.roll(smedim,-2,axis=0)
    rollxp1 = dln.roll(smedim,1,axis=1)
    rollxn1 = dln.roll(smedim,-1,axis=1)
    if minheight is not None:
        height_thresh = minheight
    else:
        height_thresh = 0.0
    peaks = ( (((smedim >= rollyn1) & (smedim > rollyp1)) |
               ((smedim > rollyn1) & (smedim >= rollyp1))) &
              (rollyp2 < hratio2*smedim) & (rollyn2 < hratio2*smedim) &
	      (smedim > height_thresh))
    
    # Now require trace heights to be within ~30% of at least one neighboring
    #   median-filtered column block
    peaksht = ((np.abs(smedim[peaks]-rollxp1[peaks])/smedim[peaks] < neifluxratio) |
              (np.abs(smedim[peaks]-rollxn1[peaks])/smedim[peaks] < neifluxratio))
    ind = np.where(peaks.ravel())[0][peaksht]
    ypind,xpind = np.unravel_index(ind,peaks.shape)
    xindex = dln.create_index(xpind)
    
    # Boolean mask image for the trace peak pixels
    pmask = np.zeros(smedim.shape,bool)
    pmask[ypind,xpind] = True
    
    # Compute height threshold from the background pixels
    #  need the traces already to do this
    if minheight is None:
        # Grow peak mask by 7 pixels in spatial and 5 in dispersion direction
        #  the rest of the pixels are background
        exclude_mask = convolve2d(pmask,np.ones((7,5)),mode='same')
        backmask = (exclude_mask < 0.5)
        backpix = smedim[backmask]
        medback,sigback = backvals(backpix)
        # Use the final background values to set the height threshold
        height_thresh = medback + minsigheight * sigback
        if verbose: print('height threshold',height_thresh)
        # Impose this on the peak mask
        #oldpmask = pmask
        #pmask = (smedim*pmask > height_thresh)
        #ypind,xpind = np.where(pmask)
        oldpmask = pmask
        ypind,xpind = np.where((smedim*pmask > height_thresh))
        # Create the X-values index of peaks
        xindex = dln.create_index(xpind)
        
    # Now match up the traces across column blocks
    # Loop over the column blocks and either connect a peak
    # to an existing trace or start a new trace
    # only connect traces that haven't been "lost"
    tracelist = []
    # Looping over unique column blocks, not every one might be represented
    for i in range(len(xindex['value'])):
        ind = xindex['index'][xindex['lo'][i]:xindex['hi'][i]+1]
        nind = len(ind)
        xind = xpind[ind]
        yind = ypind[ind]
        yind.sort()    # sort y-values
        xb = xind[0]   # this column block index        

        # Deal with neighbors
        #  we shouldn't have any neighbors in Y
        if len(ind)>1:
            diff = dln.slope(np.array(yind))
            bd, = np.where(diff == 1)
            if len(bd)>0:
                torem = []
                for j in range(len(bd)):
                    lo = bd[j]
                    hi = lo+1
                    if medim[yind[lo],xb] >= medim[yind[hi],xb]:
                        torem.append(hi)
                    else:
                        torem.append(lo)
                yind = np.delete(yind,np.array(torem))
                xind = np.delete(xind,np.array(torem))            
                nind = len(yind)

        if verbose: print(i,xb,len(xind),'peaks')
        # Adding traces
        #   Loop through all of the existing traces and try to match them
        #   to new ones in this row
        tymatches = []
        for j in range(len(tracelist)):
            itrace = tracelist[j]
            y1 = itrace['yvalues'][-1]
            x1 = itrace['xbvalues'][-1]
            deltax = xb-x1
            # Only connect traces that haven't been lost
            if deltax<3:
                # Check the pmask boolean image to see if it connects
                ymatch = None
                if pmask[y1,xb]:
                    ymatch = y1
                elif pmask[y1-1,xb]:
                    ymatch = y1-1                       
                elif pmask[y1+1,xb]:
                    ymatch = y1+1

                if ymatch is not None:
                    itrace['yvalues'].append(ymatch)
                    itrace['xbvalues'].append(xb)
                    itrace['xvalues'].append(xmed[xb])
                    itrace['heights'].append(smedim[ymatch,xb])
                    itrace['ncol'] += 1                    
                    tymatches.append(ymatch)
                    if verbose: print(' Trace '+str(j)+' Y='+str(ymatch)+' matched')
                else:
                    # Lost
                    if itrace['lost']:
                        itrace['lost'] = True
                        if verbose: print(' Trace '+str(j)+' LOST')
            # Lost, separation too large
            else:
                # Lost
                if itrace['lost']:
                    itrace['lost'] = True
                    if verbose: print(' Trace '+str(j)+' LOST')        
        # Add new traces
        # Remove the ones that matched
        yleft = yind.copy()
        if len(tymatches)>0:
            ind1,ind2 = dln.match(yind,tymatches)
            nmatch = len(ind1)
            if nmatch>0 and nmatch<len(yind):
                yleft = np.delete(yleft,ind1)
            else:
                yleft = []
        # Add the ones that are left
        if len(yleft)>0:
            if verbose: print(' Adding '+str(len(yleft))+' new traces')
        for j in range(len(yleft)):
            # Skip traces too low or high
            if yleft[j]<2 or yleft[j]>(ny-3):
                continue
            itrace = {'index':0,'ncol':0,'yvalues':[],'xbvalues':[],'xvalues':[],'heights':[],
                      'lost':False}
            itrace['index'] = len(tracelist)+1
            itrace['ncol'] = 1
            itrace['nbin'] = nbin  # save the binning information            
            itrace['yvalues'].append(yleft[j])
            itrace['xbvalues'].append(xb)
            itrace['xvalues'].append(xmed[xb])                
            itrace['heights'].append(smedim[yleft[j],xb])
            tracelist.append(itrace)
            if verbose: print(' Adding Y='+str(yleft[j]))
            
    # For each trace, check if it continues at the end, but below the nominal height threshold
    for i in range(len(tracelist)):
        itrace = tracelist[i]
        # Check lower X-values
        flag = 0
        nnew = 0
        x1 = itrace['xbvalues'][0]
        y1 = itrace['yvalues'][0]
        if x1==0 or x1==(nxb-1): continue  # at edge already
        while (flag==0):
            # Check peaks, heights must be within 30% of the height of the last one
            doesconnect = peaks[y1-1:y1+2,x1-1] & (np.abs(smedim[y1-1:y1+2,x1-1]-smedim[y1,x1])/smedim[y1,x1] < neifluxratio)
            if np.sum(doesconnect)>0:
                newx = x1-1
                if doesconnect[1]==True:
                    newy = y1
                elif doesconnect[0]==True:
                    newy = y1-1
                else:
                    newy = y1+1
                # Add new column to the trace (to beginning)
                itrace['yvalues'].insert(0,newy)
                itrace['xbvalues'].insert(0,newx)
                itrace['xvalues'].insert(0,xmed[newx])
                itrace['heights'].insert(0,smedim[newy,newx])
                itrace['ncol'] += 1
                if verbose:
                    print(' Trace '+str(i)+' Adding X='+str(newx)+' Y='+str(newy))
                # Update x1/y1
                x1 = newx
                y1 = newy
                # If at edge, stop                
                if newx==0 or newx==(nxb-1): flag = 1
                nnew += 1
            # No match
            else:
                flag = 1
        # Stuff it back in
        tracelist[i] = itrace
        
        # Check higher X-values
        flag = 0
        nnew = 0
        x1 = itrace['xbvalues'][-1]
        y1 = itrace['yvalues'][-1]
        if x1==0 or x1==(nxb-1): continue  # at edge already
        while (flag==0):
            # Check peaks, heights must be within 30% of the height of the last one
            doesconnect = peaks[y1-1:y1+2,x1+1] & (np.abs(smedim[y1-1:y1+2,x1+1]-smedim[y1,x1])/smedim[y1,x1] < neifluxratio)
            if np.sum(doesconnect)>0:
                newx = x1+1
                if doesconnect[1]==True:
                    newy = y1
                elif doesconnect[0]==True:
                    newy = y1-1
                else:
                    newy = y1+1
                # Add new column to the trace
                itrace['yvalues'].append(newy)
                itrace['xbvalues'].append(newx)
                itrace['xvalues'].append(xmed[newx])
                itrace['heights'].append(smedim[newy,newx])
                itrace['ncol'] += 1
                if verbose:
                    print(' Trace '+str(i)+' Adding X='+str(newx)+' Y='+str(newy))
                # Update x1/y1
                x1 = newx
                y1 = newy
                # If at edge, stop                
                if newx==0 or newx==(nxb-1): flag = 1
                nnew += 1
            # No match
            else:
                flag = 1      
        # Stuff it back in
        tracelist[i] = itrace
        
    # Impose minimum number of column blocks
    if mincol <= 1.0:
        mincolblocks = int(mincol*nxb)
    else:
        mincolblocks = int(mincol)
    tracelist = [tr for tr in tracelist if tr['ncol']>mincolblocks]
    if len(tracelist)==0:
        print('No traces left')
        return []
    
    # Calculate Gaussian parameters
    nprofiles = np.sum([tr['ncol'] for tr in tracelist])
    profiles = np.zeros((nprofiles,9),float) + np.nan  # all bad to start
    xprofiles = np.zeros((nprofiles,9),int)
    trindex = np.zeros(nprofiles,int)
    count = 0
    for t,tr in enumerate(tracelist):
        for i in range(tr['ncol']):
            if tr['yvalues'][i]<4:        # at bottom edge
                lo = 4-tr['yvalues'][i]
                yp = np.zeros(9,float)+np.nan
                yp[lo:] = medim[:tr['yvalues'][i]+5,tr['xbvalues'][i]]                
            elif tr['yvalues'][i]>(ny-5):   # at top edge
                hi = 9-(ny-tr['yvalues'][i])+1
                yp = np.zeros(9,float)+np.nan
                yp[:hi] = medim[tr['yvalues'][i]-4:,tr['xbvalues'][i]]
            else:
                lo = tr['yvalues'][i]-4
                hi = tr['yvalues'][i]+5
                yp = medim[lo:hi,tr['xbvalues'][i]]
            profiles[count,:] = yp
            xp = np.arange(9)+tr['yvalues'][i]-4            
            xprofiles[count,:] = xp
            trindex[count] = t
            count += 1
            
    #return xp,medim
    #return xprofiles,profiles
            
    # Get Gaussian parameters for all profiles at once
    pars = gpars(xprofiles,profiles)
    # Stuff the Gaussian parameters into the list
    count = 0
    for tr in tracelist:
        tr['gyheight'] = np.zeros(tr['ncol'],float)
        tr['gycenter'] = np.zeros(tr['ncol'],float)
        tr['gysigma'] = np.zeros(tr['ncol'],float)
        for i in range(tr['ncol']):
            tr['gyheight'][i] = pars[count,0]
            tr['gycenter'][i] = pars[count,1]
            tr['gysigma'][i] = pars[count,2]
            count += 1
        
    # Fit trace coefficients
    for tr in tracelist:
        tr['tcoef'] = None
        tr['xmin'] = np.min(tr['xvalues'])-nbin//2
        tr['xmax'] = np.max(tr['xvalues'])+nbin//2
        xt = np.array(tr['xvalues'])
        # Trace Y-position
        yt = tr['gycenter']
        norder = 3
        coef = np.polyfit(xt,yt,norder)
        resid = yt-np.polyval(coef,xt)
        std = np.std(resid)
        # Check if we need to go to 4th order
        if std>0.1:
            norder = 4
            coef = np.polyfit(xt,yt,norder)
            resid = yt-np.polyval(coef,xt)
            std = np.std(resid)
        # Check if we need to go to 5th order
        if std>0.1:
            norder = 5
            coef = np.polyfit(xt,yt,norder)
            resid = yt-np.polyval(coef,xt)
            std = np.std(resid)
        tr['tcoef'] = coef
        tr['tstd'] = std
        # Fit Gaussian sigma as well
        syt = tr['gysigma']
        norder = 2
        coef = np.polyfit(xt,syt,norder)
        resid = syt-np.polyval(coef,xt)
        std = np.std(resid)
        # Check if we need to go to 3rd order
        if std>0.05:
            norder = 3
            coef = np.polyfit(xt,syt,norder)
            resid = syt-np.polyval(coef,xt)
            std = np.std(resid)
        # Check if we need to go to 4th order
        if std>0.05:
            norder = 4
            coef = np.polyfit(xt,syt,norder)
            resid = syt-np.polyval(coef,xt)
            std = np.std(resid)
        tr['sigcoef'] = coef
        tr['sigstd'] = std

    # -resample each column onto the integer rows
    # -can fit polynomial to each row, just like Horne/Marshall method
    # -to make the model, get the values on the integer rows, and then
    #  resample onto the correct shifted values

    for t,tr in enumerate(tracelist):
        profiles = np.zeros((tr['ncol'],9),float)+np.nan
        xprofiles = np.zeros((tr['ncol'],9),float)
        dxprofiles = np.zeros((tr['ncol'],9),float)
        xprofile = np.arange(-4,5)
        yprofile = np.zeros((tr['ncol'],9),float)
        for i in range(tr['ncol']):
            if tr['yvalues'][i]<4:        # at bottom edge
                lo = 4-tr['yvalues'][i]
                yp = np.zeros(9,float)+np.nan
                yp[lo:] = medim[:tr['yvalues'][i]+5,tr['xbvalues'][i]]                
            elif tr['yvalues'][i]>(ny-5):   # at top edge
                hi = 9-(ny-tr['yvalues'][i])+1
                yp = np.zeros(9,float)+np.nan
                yp[:hi] = medim[tr['yvalues'][i]-4:,tr['xbvalues'][i]]
            else:
                lo = tr['yvalues'][i]-4
                hi = tr['yvalues'][i]+5
                yp = medim[lo:hi,tr['xbvalues'][i]]
            profiles[i,:] = yp
            xp = np.arange(9)+tr['yvalues'][i]-4
            xprofiles[i,:] = xp            
            dxp = xp-np.polyval(tr['tcoef'],tr['xvalues'][i])
            dxprofiles[i,:] = dxp
            good = np.isfinite(yp)
            yout = dln.interp(dxp[good],yp[good],xprofile,kind='quadratic',extrapolate=False)
            yprofile[i,:] = yout/np.nansum(yout)

        # Fit each row
        coefarr = np.zeros((9,3),float)
        for i in range(9):
            good = np.isfinite(yprofile[:,i])
            if np.sum(good)>0.5*tr['ncol']:
                x = np.array(tr['xvalues'])[good]
                y = yprofile[good,i]
                coef1 = np.polyfit(x,y,1)
                model1 = np.polyval(coef1,x)
                rms1 = np.sqrt(np.mean((y-model1)**2))
                chisq1 = np.sum((y-model1)**2)
                lnl1 = -0.5*np.sum((y-model1)**2)-np.pi*len(y)
                bic1 = 2*np.log(len(y))-2*lnl1
                coef2 = np.polyfit(x,y,2)
                model2 = np.polyval(coef2,x)
                rms2 = np.sqrt(np.mean((y-model2)**2))
                chisq2 = np.sum((y-model2)**2)
                lnl2 = -0.5*np.sum((y-model2)**2)-np.pi*len(y)
                bic2 = 3*np.log(len(y))-2*lnl2
                coef3 = np.polyfit(x,y,3)
                model3 = np.polyval(coef3,x)
                rms3 = np.sqrt(np.mean((y-model3)**2))
                chisq3 = np.sum((y-model2)**2)                
                lnl3 = -0.5*np.sum((y-model3)**2)-np.pi*len(y)
                bic3 = 4*np.log(len(y))-2*lnl3
                coef4 = np.polyfit(x,y,4)
                model4 = np.polyval(coef4,x)
                rms4 = np.sqrt(np.mean((y-model4)**2))
                chisq4 = np.sum((y-model4)**2)                
                lnl4 = -0.5*np.sum((y-model4)**2)-np.pi*len(y)
                bic4 = 5*np.log(len(y))-2*lnl4
                coef5 = np.polyfit(x,y,5)
                model5 = np.polyval(coef5,x)
                rms5 = np.sqrt(np.mean((y-model5)**2))
                chisq5 = np.sum((y-model5)**2)
                lnl5 = -0.5*np.sum((y-model5)**2)-np.pi*len(y)
                bic5 = 6*np.log(len(y))-2*lnl5

                #medy = median_filter(y,7)

                # Maybe use Bayesian Information Criterion (BIC)
                # to decide which order to use
                # BIC = k*ln(n) - 2*ln(L)
                # k = number of model parameters
                # n = number of data points
                # L = the maximized value of the likelihood function of the model
                #   = -0.5*chisq - 0.5*Sum(2*pi*sigma_i**2)

                from sklearn.preprocessing import PolynomialFeatures
                import statsmodels.api as sm

                model = sm.OLS(y,x).fit()
                
                from dlnpyutils import plotting as pl
                import pdb; pdb.set_trace()

            
    # Fit empirical correction terms
    for i,tr in enumerate(tracelist):
        ind, = np.where(trindex == i)
        xp1 = np.array(tr['xvalues']).reshape(-1,1) + np.zeros(9)
        yp1 = xprofiles[ind,:]
        flx1 = profiles[ind,:]
        # we should use the polynomial values
        gfit = np.zeros(xp1.shape,float)
        for j in range(tr['ncol']):
            gp = [tr['gyheight'][j],tr['gycenter'][j],tr['gysigma'][j]]
            gfit[j,:] = dln.gaussian(yp1[j,:],*gp)
        ycenter = np.polyval(tr['tcoef'],tr['xvalues'])
        sigma = np.polyval(tr['sigcoef'],tr['xvalues'])
        hcoef = np.polyfit(tr['xvalues'],tr['gyheight'],3)
        height = np.polyval(hcoef,tr['xvalues'])
        #height = 1/(sigma*np.sqrt(2*np.pi))
        tot1 = np.sum(flx1,axis=1)
        yp2 = yp1-ycenter.reshape(-1,1)
        yp3 = yp2 / sigma.reshape(-1,1)
        flx2 = flx1 / height.reshape(-1,1)
        flx2 = flx1 / tot1.reshape(-1,1)        

        from scipy.stats import binned_statistic,binned_statistic_2d
        ybins = np.arange(-4.0,4.5,0.5)
        ymn,yedge,binnumber = binned_statistic(yp2.ravel(),flx2.ravel(),bins=ybins,statistic='mean')
        # use least squares to get slope and zero-point in each Y-bin
        # slope = Sum((xi-xmean)*(yi-ymean))/Sum((xi-xmean)^2)
        # zpterm = ymean - slope*xmean
        xmn,yedge,binnumber = binned_statistic(yp2.ravel(),xp1.ravel(),bins=ybins,statistic='mean')
        c0 = np.zeros(len(ybins))
        c1 = np.zeros(len(ybins))
        for k in range(len(ybins)):
            ind, = np.where(binnumber==k)
            x1 = xp1.ravel()[ind]
            y1 = flx2.ravel()[ind]
            xmn = np.mean(x1)
            ymn = np.mean(y1)
            slp = np.sum((x1-xmn)*(y1-ymn))/np.sum((x1-xmn)**2)
            zp = ymn-slp*xmn
            c0[k] = zp
            c1[k] = slp

        model = c0.reshape(-1,1) + x*c1.reshape(-1,1)
        
        from dlnpyutils import plotting as pl
        #pl.scatter(xp1,yp2,flx2)
        #pl.scatter(yp2,flx2,xp1)
        #pl.scatter(xp1,yp2,flx1/gfit)
        #pl.scatter(yp2,flx1/gfit,xp1)
        
        import pdb; pdb.set_trace()


        
    # Should we check for overlap of traces?
        
    # Add YMED and sort
    for i in range(len(tracelist)):
        itrace = tracelist[i]
        itrace['ymed'] = np.median(itrace['yvalues'])
        tracelist[i] = itrace
    # Sort them
    tracelist.sort(key=lambda t:t['ymed'])
        
    return tracelist
