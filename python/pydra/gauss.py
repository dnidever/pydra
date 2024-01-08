#!/usr/env python

# Imports
import numpy as np
from numba import njit
from dlnpyutils import utils as dln,robust

# Functions for estimating Gaussian parameters of peaks

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
def peakboundary(y,cen):
    """
    Find the boundary of the peak.
    Can be where the values start to increase again, or
    hit the edge of the array

    Parameters
    ----------
    y : numpy array
       Numpy array of the peak flux array.
    cen : float
       Estimate of the peak center.

    Returns
    -------
    low : int
       Index of the lower edge of the peak.
    high : int
       Index of the upper edge of the peak.

    Example
    -------

    low,high = peakboundary(y,cen)

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
    
    cen = gparscenpeak(y)

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
def gparssigmasearch_solve(u,y,sigma):
    # Scale y to get Amp and offset
    yf = gaussian(u,np.array([1.0,0.0,sigma]))
    # solve for Amp and offset
    amp,off = linregression(yf,y)
    model = yf*amp+off
    residsq = (y-model)**2
    chisq = np.sum(residsq)
    return amp,off,chisq

@njit
def gparssigmasearch(u,y,sigma0):
    """ 
    Search a region of sigma finding the best chiqsq value.

    Parameters
    ----------
    u : numpy array
       Numpy array of the shifted x-values, u=|x-cen|.
         NOT scaled by sigma.
    y : numpy array
       Numpy array of flux values.
    sigma0 : float
       Initial sigma estimate.

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
    
    amp,sigma,offset,chisq = gparssigmasearch(u,y,3.0)

    """
                  
    # Pretty fast and robust
    # 22 microseconds
    
    # Assume u extends to 3*sigma
    usq = u**2
    umax = np.max(np.abs(u))
    #sigma0 = umax/3.0

    # Initialize large arrays to hold the chisq and sigma values
    sigmaarr = np.zeros(100,float)    
    chisqarr = np.zeros(100,float)
    count = 0

    # Get chisq of initial sigma value
    amp0,off0,chisq0 = gparssigmasearch_solve(u,y,sigma0)
    sigmaarr[count] = sigma0
    chisqarr[count] = chisq0
    count += 1
    
    # Find inner edge where chisq starts to increase
    sigma = sigma0*0.75
    last_chisq = chisq0
    flag = 0
    while (flag==0):
        amp,off,chisq = gparssigmasearch_solve(u,y,sigma)
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
        amp,off,chisq = gparssigmasearch_solve(u,y,sigma)
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
    dum = np.zeros(1,float)  # need to do this otherwise numba will crash
    dum[0] = sigma
    sigma = dum[0]
    
    # Now get amp and offset for the interpolated sigma value
    amp,offset,chisq = gparssigmasearch_solve(u,y,sigma)

    # Make sure to take the best chisq solution
    if chisqarr[bestind] < chisq:
        sigma = sigmaarr[bestind]
        amp,offset,chisq = gparssigmasearch_solve(u,y,sigma)
            
    return amp,sigma,offset,chisq


@njit
def gparssearch(x,y,cen0):
    """ 
    Search a region of center and sigma finding the best chiqsq value.

    Parameters
    ----------
    x : numpy array
       Numpy array of the shifted x-values.
    y : numpy array
       Numpy array of flux values.
    cen0 : float
       Initial estimate for center.

    Returns
    -------
    amp : float
       The Gaussian amplitude value.
    center : float
       The Gaussian center value.
    sigma : float
       The Gaussian sigma value.
    offset : float
       The constant offset.
    chisq : float
       The best chisq value.

    Example
    -------
    
    amp,center,sigma,offset,chisq = gparssearch(x,y,5.0)

    """

    dx = x[1]-x[0]
    
    # Assume u extends to 3*sigma
    umax = np.max(np.abs(x-cen0))
    sig0 = umax/3.0

    # Initialize large arrays to hold the chisq and sigma values
    cenarr = np.zeros(100,float)
    sigmaarr = np.zeros(100,float)    
    chisqarr = np.zeros(100,float)
    count = 0

    # Get chisq of initial cen value
    u0 = np.abs(x-cen0)
    amp0,sigma0,off0,chisq0 = gparssigmasearch(u0,y,sig0)
    cenarr[count] = cen0
    sigmaarr[count] = sigma0    
    chisqarr[count] = chisq0
    count += 1
    
    # Find inner edge where chisq starts to increase
    step = 0.1*dx
    cen = cen0-step
    last_chisq = chisq0
    flag = 0
    while (flag==0):
        u = np.abs(x-cen)
        amp,sigma,off,chisq = gparssigmasearch(u,y,sigma0)
        sigmaarr[count] = sigma
        cenarr[count] = cen      
        chisqarr[count] = chisq
        count += 1
        if chisq < last_chisq:
            cen -= step
        else:
            flag = 1
        last_chisq = chisq
    low_cen = cen

    # Find upper edge where chisq starts to increase
    cen = cen0+step
    last_chisq = chisq0
    flag = 0
    while (flag==0):
        u = np.abs(x-cen)
        amp,sigma,off,chisq = gparssigmasearch(u,y,sigma0)
        sigmaarr[count] = sigma
        cenarr[count] = cen   
        chisqarr[count] = chisq
        count += 1
        if chisq < last_chisq:
            cen += step
        else:
            flag = 1
        last_chisq = chisq
    high_cen = cen
    
    # Trim the chisq/sigma values
    sigmaarr = sigmaarr[0:count]
    cenarr = cenarr[0:count]    
    chisqarr = chisqarr[0:count]
    # sort them
    si = np.argsort(cenarr)
    sigmaarr = sigmaarr[si]
    cenarr = cenarr[si]    
    chisqarr = chisqarr[si]
    bestind = np.argmin(chisqarr)
    
    # Fit quadratic equation to best value and neighboring points
    center = quadratic_bisector(cenarr[bestind-1:bestind+2],
                                chisqarr[bestind-1:bestind+2])
    sigmabest = sigmaarr[bestind]

    # Now get amp, sigma and offset for interpolated center value
    u = np.abs(x-cen)
    amp,sigma,offset,chisq = gparssigmasearch(u,y,sigmabest)   

    # Make sure to take the best chisq solution
    if chisqarr[bestind] < chisq:
        center = cenarr[bestind]
        sigma = sigmaarr[bestind]
        u = np.abs(x-center)
        amp,offset,chisq = gparssigmasearch_solve(u,y,sigma)
    
    return amp,center,sigma,offset,chisq

@njit
def gparscenbisector(x,y,cen):
    """
    Use the symmetry of the curve to find the center

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
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
    
    cen = gparscenbisector(x,y,cen)

    """

    n = len(y)
    # Central pixel to start with
    xstart = int(np.ceil(cen))

    # Find the bisector of the peak
    # For each pixel on one side, find the expected position at the same
    # flux level on the other side
    ind = np.arange(n)
    left = (ind < cen)
    xleft = x[left]
    xright = x[~left]
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

    # This takes about 145 microseconds for 10 points

    dx = x[1]-x[0]
    yp = np.maximum(y,0)
    good, = np.where(np.isfinite(y))
    xp = x[good]
    yp = yp[good]
    ymax = np.max(yp)
    ymin = np.min(yp)
    xmax = xp[np.argmax(yp)]
    
    # Use peak to get first estimate of center
    xcen0 = gparscenpeak(yp)

    # Get initial range using peakboundary
    lo,hi = peakboundary(yp,int(np.round(xcen0)))
    
    # Get improved center from bisector
    cen0 = gparscenbisector(xp[lo:hi],yp[lo:hi],xcen0)

    # Get initial sigma from second moment 
    totyp = np.sum(yp[lo:hi])                                    # 0th moment
    sigma0 = np.sqrt(np.sum(y[lo:hi]*(x[lo:hi]-cen0)**2)/totyp)  # 2nd central moment

    # Get good pixels and
    #  limit to the peakboundary() lo:hi range
    index = np.arange(len(x))
    good = ((np.abs(x-cen0) <= 3*sigma0) & np.isfinite(y) &
            (index >= lo) & (index <= hi))
        
    xp = x[good]
    yp = y[good]
    yp = np.maximum(yp,0)
    u0 = (xp-cen0)**2

    # Use sigma + center search
    amp,center,sigma,offset,chisq = gparssearch(xp,yp,cen0)

    # Check the solution with no offset
    logcoef = gparslog(xp,yp)
    logchisq = np.sum((yp-gaussian(xp,logcoef))**2)

    # Use the best solution
    coef = np.zeros(4,float)
    # search method
    if chisq < logchisq:
        coef[0] = amp
        coef[1] = center
        coef[2] = sigma
        coef[3] = offset
    # Log method, no offset
    else:
        # If the log method with no offset is best, then
        # maybe the offset is small
        # Try to estimate amp and offset with the
        # log method sigma
        yf = gaussian(xp,np.array([1.0,logcoef[1],logcoef[2]]))
        # solve for Amp and offset
        amp,off = linregression(yf,yp)
        model = yf*amp+off
        chisqoff = np.sum((y-model)**2)
        # Small offset better
        if chisqoff < logchisq:
            coef[0] = amp
            coef[1] = logcoef[1]
            coef[2] = logcoef[2]
            coef[3] = off
        # No offset better
        else:
            coef[0] = logcoef[0]
            coef[1] = logcoef[1]
            coef[2] = logcoef[2]
            coef[3] = 0                

    return coef

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


