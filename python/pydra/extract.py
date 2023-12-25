# Packages that allow us to get information about objects:
import os
import numpy as np
from astropy.table import Table
from dlnpyutils import utils as dln,robust

# Astropy tools:
from astropy.io import fits
from astropy.time import Time
from doppler.spec1d import Spec1D
from scipy.ndimage import median_filter,generic_filter
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from numba import njit
from . import utils

import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import plotting as pl


# Ignore these warnings
import warnings
warnings.filterwarnings("ignore", message="OptimizeWarning: Covariance of the parameters could not be estimated")


import inspect
import sys

def recompile_nb_code():
    this_module = sys.modules[__name__]
    module_members = inspect.getmembers(this_module)

    for member_name, member in module_members:
        if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
            member.recompile()

def nanmedfilt(x,size,mode='reflect'):
    return generic_filter(x, np.nanmedian, size=size)

def findpeaks(flux):
    """ Find the peaks."""
    maxind, = argrelextrema(flux, np.greater)  # maxima
    return maxind
    
def profilefit(x,y,cenlimits=None,siglimits=None):
    """ Fit a spectral profile."""
    flux = np.sum(np.maximum(y,0))
    xmean = np.sum(x*np.maximum(y,0))/flux
    xsig = np.sqrt(np.sum((x-xmean)**2 * np.maximum(y,0)/flux))
    xsig = np.maximum(xsig,0.1)
    # Fit binned Gaussian
    p0 = [np.max(y),xmean,xsig,0.0]
    bnds = [np.array([p0[0]*0.5,xmean-1,0.5*xsig,-0.3*p0[0]]),
            np.array([p0[0]*2,xmean+1,2*xsig,0.3*p0[0]])]
    if cenlimits is not None:
        bnds[0][1] = cenlimits[0]
        bnds[1][1] = cenlimits[1]
        p0[1] = np.mean(cenlimits)
    if siglimits is not None:
        bnds[0][2] = siglimits[0]
        bnds[1][2] = siglimits[1]
        p0[2] = np.mean(siglimits)        
    if np.sum((bnds[0][:] >= bnds[1][:]))>0:
        print('problem in profilefit')
        import pdb; pdb.set_trace()

    try:
        pars,cov = dln.gaussfit(x,y,initpar=p0,bounds=bnds,binned=True)
        perror = np.sqrt(np.diag(cov))
        return pars,perror        
    except:
        print('profilefit exception')
        return None,None
    

def tracing(im,err,ytrace=None,step=15,nbin=25):
    """ Trace a spectrum. Assumed to be in the horizontal direction."""
    ny,nx = im.shape
    y,x = np.arange(ny),np.arange(nx)

    if ytrace is not None:
        if np.array(ytrace).size>1:
            ymid = np.mean(ytrace)
        else:
            ymid = ytrace
            
    # Find ymid if no trace input
    if ytrace is None:
        #tot = np.nanmedian(np.maximum(im[:,nx//2-50:nx//2+50],0),axis=1)
        tot = np.nanmedian(np.maximum(im,0),axis=1)        
        ymid = np.argmax(tot)
        
    # Trace using binned profiles, starting at middle
    nsteps = (nx//2)//step
    #if nsteps % 2 == 0: nsteps-=1
    lasty = ymid
    lastsig = 1.0
    xmnarr = []
    yhtarr = []    
    ymidarr = []
    ysigarr = []
    # Forwards
    for i in range(nsteps):
        xmn = nx//2 + i*step
        xlo = xmn - nbin//2
        xhi = xmn + nbin//2 + 1
        profile = np.nanmedian(im[:,xlo:xhi],axis=1)
        profileerr = np.nanmedian(err[:,xlo:xhi],axis=1)        
        profile[~np.isfinite(profile)] = 0.0
        profileerr[~np.isfinite(profileerr) | (profileerr<=0)] = 1e30
        flux = np.nansum(np.maximum(profile,0))
        snr = np.nanmax(profile/profileerr)  # max S/N
        if flux <= 0 or snr<5:
            continue
        ylo = np.maximum(int(np.floor(lasty-3.0*lastsig)),0)
        yhi = np.minimum(int(np.ceil(lasty+3.0*lastsig)),ny)
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        if np.sum(~np.isfinite(profileclip))>0:
            continue
        if len(yclip)==0:
            print('no pixels')
            import pdb; pdb.set_trace()
        # Limit central position using input ytrace
        #  and limit sigma
        if ytrace is not None and np.array(ytrace).size>1:
            ycen = np.mean(ytrace[xlo:xhi])
            cenlimits = [ycen-0.25,ycen+0.25]
        else:
            cenlimits = None
        siglimits = [0.4,0.6]
        pars,perror = profilefit(yclip,profileclip,cenlimits=cenlimits,siglimits=siglimits)
        if pars is None:
            continue
        xmnarr.append(xmn)
        yhtarr.append(pars[0])        
        ymidarr.append(pars[1])
        ysigarr.append(pars[2])
        # Remember
        lasty = pars[1]
        lastsig = pars[2]
        
    # Backwards
    lasty = ymid
    lastsig = 0.5
    for i in np.arange(1,nsteps):
        xmn = nx//2 - i*step
        xlo = xmn - nbin//2
        xhi = xmn + nbin//2 + 1
        profile = np.nanmedian(im[:,xlo:xhi],axis=1)
        profileerr = np.nanmedian(err[:,xlo:xhi],axis=1)        
        profile[~np.isfinite(profile)] = 0.0
        profileerr[~np.isfinite(profileerr) | (profileerr<=0)] = 1e30
        flux = np.nansum(np.maximum(profile,0))
        snr = np.nanmax(profile/profileerr)  # max S/N
        if flux <= 0 or snr<5:
            continue
        ind = np.argmax(profile)
        ylo = np.maximum(int(np.floor(lasty-3.0*lastsig)),0)
        yhi = np.minimum(int(np.ceil(lasty+3.0*lastsig)),ny)
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        if np.sum(~np.isfinite(profileclip))>0:
            continue        
        if len(yclip)==0:
            print('no pixels')
            import pdb; pdb.set_trace()
        # Limit central position using input ytrace
        #  and limit sigma
        if ytrace is not None and np.array(ytrace).size>1:
            ycen = np.mean(ytrace[xlo:xhi])
            cenlimits = [ycen-0.25,ycen+0.25]
        else:
            cenlimits = None
        siglimits = [0.4,0.6]
        pars,perror = profilefit(yclip,profileclip,cenlimits=cenlimits,siglimits=siglimits)        
        if pars is None:
            continue        
        xmnarr.append(xmn)
        yhtarr.append(pars[0])        
        ymidarr.append(pars[1])
        ysigarr.append(pars[2])
        # Remember
        lasty = pars[1]
        lastsig = pars[2]

    ttab = Table((xmnarr,ymidarr,ysigarr,yhtarr),names=['x','y','ysig','amp'])
    ttab.sort('x')
        
    return ttab

def optimalpsf(im,ytrace,err=None,off=10,backoff=50,smlen=31):
    """ Compute the PSF from the image using "optimal extraction" techniques."""
    ny,nx = im.shape
    yest = np.nanmedian(ytrace)
    # Get the subimage
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    # Background subtract
    med = np.nanmedian(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    suberr = imerr[yblo:ybhi,:]
    # Make sure the arrays are float64
    subim = subim.astype(float)
    suberr = suberr.astype(float)    
    # Mask other parts of the image
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    sim = subim*mask
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

def extract(image,errim=None,kind='psf',recenter=False,skyfit=False):
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
         boxcar, psf, optional, perfectionism
    recenter : bool, optional
       Recenter the PSF on the image.  Default is False.
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
    mask = self.mask(nsigma=5)
    xr = [slc[1].start,slc[1].stop]
    yr = [slc[0].start,slc[0].stop]
    im = image[slc]
    if errim is not None:
        err = errim[slc]
    else:
        err = None
    psf = self.model(xr=xr,yr=yr)
    # Recenter
    if recenter:
        sh = im.shape
        xhalf = sh[1]//2
        medim = np.median(im[:,xhalf-50:xhalf+50],axis=1)
        medpsf = np.median(psf[:,xhalf-50:xhalf+50],axis=1)
        import pdb; pdb.set_trace()

    # Do the extraction
    # -- Boxcar --
    if kind=='boxcar':
        boxflux = np.nansum(mask*im,axis=0)
        if err is not None:
            boxerr = np.sqrt(np.nansum(mask*err**2,axis=0))
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
        out = extract_optimal(im,ytrace,imerr=err,verbose=False,
                              off=10,backoff=50,smlen=31)
        flux,fluxer,trace,psf = out
        dt = [('flux',float),('err',float),('sky',float),('skyerr',float)]
        tab = np.zeros(len(flux),dtype=np.dtype(dt))
        tab['flux'] = flux
        tab['err'] = fluxerr
        tab['sky'] = sky
        tab['skyerr'] = skyerr
            
    # -- PSF extraction --
    elif kind=='psf':
        out = extract.extract_psf(im,psf,err=err,skyfit=skyfit)
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
        out = extract_optimal(im,ytrace,imerr=None,verbose=False,off=10,backoff=50,smlen=31)
        flux,fluxerr,trace,psf = out
        dt = [('flux',float),('err',float)]
        tab = np.zeros(len(flux),dtype=np.dtype(dt))
        tab['flux'] = flux
        tab['err'] = fluxerr            
            
    return tab

    
def extract_optimal(im,ytrace,imerr=None,verbose=False,off=10,backoff=50,smlen=31):
    """ Extract a spectrum using optimal extraction (Horne 1986)"""
    # Make fake error image
    if imerr is None:
        imerr = np.sqrt(np.maximum(im*10000/np.max(im),1))
    ny,nx = im.shape
    yest = np.nanmedian(ytrace)
    # Get the subo,age
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    if nback < 0:
        print('problems in extract_optimal')
        return None,None,None,None
    # Background subtract
    med = np.nanmedian(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    suberr = imerr[yblo:ybhi,:]
    # Make sure the arrays are float64
    subim = subim.astype(float)
    suberr = suberr.astype(float)    
    # Mask other parts of the image
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    sim = subim*mask
    serr = suberr*mask
    badpix = (serr <= 0)
    serr[badpix] = 1e20
    # Compute the profile/probability matrix from the image
    tot = np.nansum(np.maximum(sim,0),axis=0)
    tot[(tot<=0) | ~np.isfinite(tot)] = 1
    psf1 = np.maximum(sim,0)/tot
    psf = np.zeros(psf1.shape,float)
    for i in range(nback):
        psf[i,:] = nanmedfilt(psf1[i,:],smlen)
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    totpsf = np.nansum(psf,axis=0)
    totpsf[(totpsf<=0) | (~np.isfinite(totpsf))] = 1
    psf /= totpsf
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    # Compute the weights
    wt = psf**2/serr**2
    wt[(wt<0) | ~np.isfinite(wt)] = 0
    totwt = np.nansum(wt,axis=0)
    badcol = (totwt<=0)
    totwt[badcol] = 1
    # Compute the flux and flux error
    flux = np.nansum(psf*sim/serr**2,axis=0)/totwt
    fluxerr = np.sqrt(1/totwt)    
    fluxerr[badcol] = 1e30  # bad columns
    # Recompute the trace
    trace = np.nansum(psf*yy,axis=0)+yblo
    
    # Check for outliers
    diff = (sim-flux*psf)/serr
    bad = (diff > 25)
    if np.nansum(bad)>0:
        # Mask bad pixels
        sim[bad] = 0
        serr[bad] = 1e20
        # Recompute the flux
        wt = psf**2/serr**2
        totwt = np.nansum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1        
        flux = np.nansum(psf*sim/serr**2,axis=0)/totwt
        fluxerr = np.sqrt(1/totwt)
        fluxerr[badcol] = 1e30  # bad columns
        # Recompute the trace
        trace = np.nansum(psf*yy,axis=0)+yblo

    # Need at least ONE good profile point to measure a flux
    ngood = np.sum((psf>0.01)*np.isfinite(im),axis=0)
    badcol = (ngood==0)
    flux[badcol] = 0.0
    fluxerr[badcol] = 1e30 
        
    return flux,fluxerr,trace,psf


def extract_psf(im,psf,err=None,skyfit=True):
    """ Extract spectrum with a PSF."""

    if err is None:
        err = np.ones(im.shape,float)
    # Fit the sky
    if skyfit:
        # Compute the weights
        # If you are solving for flux and sky, then
        # you need to do 1/err**2 weighting
        wt = 1/err**2
        wt[(wt<0) | ~np.isfinite(wt)] = 0
        totwt = np.sum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1
        # Perform weighted linear regression
        flux,fluxerr,sky,skyerr = utils.weightedregression(psf,im,wt,zero=False)
        # Compute the flux and flux error
        fluxerr[badcol] = 1e30  # bad columns
        # Need at least ONE good profile point to measure a flux
        ngood = np.sum((psf>0.01)*np.isfinite(im),axis=0)
        badcol = (ngood==0)
        flux[badcol] = 0.0
        fluxerr[badcol] = 1e30
        
        return flux,fluxerr,sky,skyerr        
        
    # Only solve for flux
    #  assume sky was already subtracted
    else:
        wt = psf**2/err**2
        totwt = np.sum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1
        # Perform weighted linear regression
        flux,fluxerr = utils.weightedregression(psf,im,wt,zero=True)
        # Compute the flux and flux error
        fluxerr[badcol] = 1e30  # bad columns
        # Need at least ONE good profile point to measure a flux
        ngood = np.sum((psf>0.01)*np.isfinite(im),axis=0)
        badcol = (ngood==0)
        flux[badcol] = 0.0
        fluxerr[badcol] = 1e30
        
        return flux,fluxerr

def extractcol(im,err,psf):
    # Optimal extraction of a single column
    wt = psf**2/err**2
    wt[(wt<0) | ~np.isfinite(wt)] = 0
    totwt = np.nansum(wt,axis=0)
    if totwt <= 0: totwt=1
    # Compute the flux and flux error
    flux = np.nansum(psf*im/err**2,axis=0)/totwt
    fluxerr = np.sqrt(1/totwt)
    if np.isfinite(flux)==False:
        fluxerr = 1e30  # bad columns
    return flux,fluxerr

def findworstcolpix(im,err,psf):
    """ Find worst outlier pixel in a column"""
    n = len(im)
    gd = np.arange(n)    
    # Loop over the good pixels and take on away each time
    #  then recompute the flux and rchisq
    rchiarr = np.zeros(n)
    fluxarr = np.zeros(n)
    fluxerrarr = np.zeros(n)
    for j in range(n):
        ind = gd.copy()
        ind = np.delete(ind,j)
        tflx,tflxerr = extractcol(im[ind],err[ind],psf[ind])
        fluxarr[j] = tflx
        fluxerrarr[j] = tflxerr
        rchi1 = np.sum((im[ind]-tflx*psf[ind])**2/err[ind]**2)/(n-1)
        rchiarr[j] = rchi1
    bestind = np.argmin(rchiarr)
    bestrchi = rchiarr[bestind]
    bestflux = fluxarr[bestind]
    bestfluxerr = fluxerrarr[bestind]
    return bestind,bestrchi,bestflux,bestfluxerr

def fixbadpixels(im,err,psf):
    """ Fix outlier pixels using the PSF."""

    ny,nx = im.shape
    # Compute chi-squared for each column and check outliers for bad pixels
    mask = (psf > 0.01)
    # Recompute the flux
    wt = psf**2/err**2
    totwt = np.nansum(wt,axis=0)
    badcol = (totwt<=0)
    totwt[badcol] = 1        
    flux = np.nansum(psf*im/err**2,axis=0)/totwt
    # Calculate model and chisq
    model = psf*flux.reshape(1,-1)
    chisq = np.sum((model-im*mask)**2/err**2,axis=0)
    ngood = np.sum(np.isfinite(mask*im)*mask,axis=0)
    rchisq = chisq/np.maximum(ngood,1)
    #smlen = np.minimum(7,nx)
    #medrchisq = nanmedfilt(rchisq,smlen,mode='mirror')
    #sigrchisq = dln.mad(rchisq-medrchisq)
    medrchisq = np.maximum(np.nanmedian(rchisq),1.0)
    sigrchisq = dln.mad(rchisq-medrchisq,zero=True)
    coltofix, = np.where(((rchisq-medrchisq > 5*sigrchisq) | ~np.isfinite(rchisq)) & (rchisq>5) & (ngood>0))

    fixmask = np.zeros(im.shape,bool)
    fixim = im.copy()
    fixflux = flux.copy()
    fixfluxerr = flux.copy()    
    
    # Loop over columns to try to fix
    for i,c in enumerate(coltofix):
        cim = im[:,c]
        cerr = err[:,c]
        cpsf = psf[:,c]
        cflux = flux[c]
        gd, = np.where(mask[:,c]==True)
        ngd = len(gd)
        rchi = np.sum((cim[gd]-cflux*cpsf[gd])**2/cerr[gd]**2)/ngd

        # We need to try each pixel separately because if there is an
        # outlier pixel then the flux will be bad and all pixels will be "off"
        
        # While loop to fix bad pixels
        prevrchi = rchi
        count = 0
        fixed = True
        while ((fixed==True) & (ngd>2)):
            # Find worse outlier pixel in a column
            bestind,bestrchi,bestflux,bestfluxerr = findworstcolpix(cim[gd],cerr[gd],cpsf[gd])
            # Make sure it is a decent improvement
            fixed = False
            if bestrchi<0.8*prevrchi and prevrchi>5:
                curfixind = gd[bestind]
                fixmask[curfixind,c] = True
                #print('Fixing pixel ['+str(curfixind)+','+str(c)+'] ',prevrchi,bestrchi)
                # Find current and previous fixed pixels
                #  need to replace all of their values
                #  using this new flux
                fixind, = np.where(fixmask[:,c]==True)
                fixim[fixind,c] = cpsf[fixind]*bestflux
                fixflux[c] = bestflux
                fixfluxerr[c] = bestfluxerr
                gd = np.delete(gd,bestind)  # delete the pixel from the good pixel list
                ngd -= 1
                fixed = True
                prevrchi = bestrchi
            count += 1            
            
    return fixim,fixmask,fixflux,fixfluxerr


def fix_outliers(im,err=None,nsigma=5,nfilter=11,niter=3):
    """ Fix large outlier pixels in an image."""
    ny,nx = im.shape
    outim = im.copy()
    if err is not None:
        outerr = err.copy()
        mederr = np.nanmedian(err[im!=0])
        if np.isfinite(mederr)==0:
            mederr = np.nanmedian(err)
        outerr[~np.isfinite(outerr)] = mederr
    med = np.nanmedian(im[im!=0])
    if np.isfinite(med)==0:
        med = 0.0
    outim[~np.isfinite(outim)] = med
    for i in range(niter):
        for j in range(ny):
            line = outim[j,:]
            nonzero = (line != 0.0)
            if np.sum(nonzero)==0:
                continue        
            filt = median_filter(line,nfilter)        
            diff = line-filt
            sig = dln.mad(diff[nonzero])
            bad = (np.abs(diff) > nsigma*sig)
            if np.sum(bad)>0:
                outim[j,bad] = filt[bad]
                outerr[j,bad] *= 10  # increase the err by 10
    if err is not None:
        return outim,outerr
    else:
        return outim
        

def epsfmodel(epsf,spec,xcol,yrange=[0,2048]):
    """ Create model image using EPSF and best-fit values."""
    # spec [Ntrace], best-fit flux values
    ntrace = len(epsf)
    if fibers is None:
        fibers = np.arange(ntrace)
    
    # Create the Model 2D image
    if yrange is not None:
        model = np.zeros(yrange[1]-yrange[0],float)
        ylo = yrange[0]
    else:
        ylo = 0
        model = np.zeros(2048,float)
    flx = np.copy(spec)
    bad = (flx<=0)
    if np.sum(bad)>0:
        flx[bad] = 0
    for k in range(len(epsf)):
        p1 = epsf[k]
        lo = epsf[k]['ylo']
        hi = epsf[k]['yhi']
        xlo = epsf[k]['xlo']
        xindpsf = xcol-xlo
        psf = p1['psf'][:,xindpsf]
        model[lo-ylo:hi+1-ylo] += psf*flx[k]
    return model


def extract(im,err,tr,doback=False):
    """
    This extracts flux of multiple spectra using empirical PSFs.

    Extract spectrum under the assumption that a given pixel only contributes
    to two neighboring traces, leading to a tridiagonal matrix inversion.

    Parameters
    ----------
    im : numpy array
       The 2D flux column.
    err : numpy array
       The 2D uncertainty column.
    tr : Traces object
       Traces object.
    doback : boolean, optional
       Subtract the background.  False by default.

    Returns
    -------
    outstr : dict
        The 1D output structure with FLUX, VAR and MASK.
    back : numpy array
        The background
    model : numpy array
        The model 2D image

    Example
    -------

    outstr,back,model = extractcol(flux,fluxerr,xcol,tr)

    By J. Holtzman  2011
    Translated to Python  D.Nidever May 2011
    """
    
    var = np.copy(err**2)
    ny,nx = im.shape
    
    # Loop over the columns
    for i in range(nx):
        xcol = i
        flux = im[:,xcol]
        fluxerr = err[:,xcol]
        
        # Only keep traces that overlap this column
        index = []
        ysize = 0
        for i in range(len(tr)):
            t = tr[i]
            if t.xmin<=xcol and t.xmax>=xcol:
                index.append(i)
                ymid = np.polyval(t.coef,xcol)
                sig = np.polyval(t.sigcoef,xcol)
                ylo = int(np.floor(ymid-5*sig))
                yhi = int(np.ceil(ymid+5*sig))
                ny = yhi-ylo+1
                ysize = np.maximum(ysize,ny)
        # Now get the PSF information
        ylo = np.zeros(len(index),int)
        yhi = np.zeros(len(index),int)
        psf = np.zeros((len(index),ysize),float)
        for i in range(len(index)):
            t = tr[index[i]]
            ymid = np.polyval(t.coef,xcol)
            sig = np.polyval(t.sigcoef,xcol)
            ylo1 = int(np.floor(ymid-5*sig))
            yhi1 = int(np.ceil(ymid+5*sig))
            psf1 = t.model(xr=[xcol,xcol],yr=[ylo1,yhi1])
            psf1 = psf1[:,0]
            ylo[i] = ylo1
            yhi[i] = yhi1
            psf[i,:len(psf1)] = psf1
        
        ntrace = len(index)
        if ntrace==0:
            continue

        # Only solve traces that OVERLAP

        import pdb; pdb.set_trace()
        
        # Solve for the fluxes
        spec,specerr,mask,back = _extractcol(flux,fluxerr,ylo,yhi,psf,doback=doback)
    
        # Initialize output arrays
        dt = [('flux',float),('err',float),('mask',int),('back',float),('trace',int)]
        out = np.zeros(ntrace,dtype=np.dtype(dt))
        out['flux'] = spec
        out['err'] = specerr
        out['mask'] = mask
        out['back'] = back
        out['trace'] = np.array(index)+1
        
        # Create the Model 2D image
        model = psfmodelcol(ylo,yhi,psf,spec,back)


        import pdb; pdb.set_trace()

        
    return out,model


def psfmodelcol(ylo,yhi,psf,spec,back,yrange=None):
    """ Create model image using EPSF and best-fit values."""
    # spec [Ntrace], best-fit flux values
    ntrace = len(ylo)
    # Create the Model 2D image
    if yrange is not None:
        model = np.zeros(yrange[1]-yrange[0],float)
        y0 = yrange[0]
    else:
        y0 = 0
        model = np.zeros(np.max(yhi)+5,float)
    flx = np.copy(spec)
    bad = (flx<=0)
    if np.sum(bad)>0:
        flx[bad] = 0
    for k in range(ntrace):
        ylo1 = ylo[k]
        yhi1 = yhi[k]
        psf1 = np.copy(psf[k,:yhi1-ylo1])
        model[ylo1-y0:yhi1-y0] += spec[k]*psf1
    model += back
    return model

def _solvecol(flux,fluxerr,psf,doback=True):
    """
    Solve single column with full lineast least squares.

    Parameters
    ----------
    flux : numpy array
       The 1D flux column.
    fluxerr : numpy array
       The 1D uncertainty column.
    psf : numpy array
       PSF model for each trace for the full data length.
    doback : boolean, optional
       Subtract the background.  True by default.

    Returns
    -------
    spec : numpy array
       Final flux values.
    specerr : numpy array
       Uncertainties in spec.
    back : float
       Background value.
    backerr : float
       Uncertainty in background value.

    Example
    -------

    spec,specerr,back,backerr = _solvecol(flux,fluxerr,psf,doback=True)

    """

    # Set WEIGHT of BAD data to 0
    # we could also remove "bad" elements from flux/fluxerr/weight/psf
        
    # Solve it
    #  use weighted linear least squares
    A = psf
    if doback:  # add constant
        A = np.hstack((A,np.ones((psf.shape[0],1),float)))
    B = flux.copy()
    # When solving it with lstsq(), we need to rescale A and B by sqrt(weight)
    weight = 1/fluxerr**2
    wtsqr = np.sqrt(weight)
    bad = ((~np.isfinite(flux)) | (flux<0) | (~np.isfinite(fluxerr)))
    if np.sum(bad)>0:
        B[bad] = 0
        wtsqr[bad] = 0
    Aw = A*wtsqr.reshape(-1,1)
    Bw = (B*wtsqr).reshape(-1,1)
    x,resid,rank,s = np.linalg.lstsq(Aw, Bw)
    model = np.dot(A,x[:,0])
    chisq = np.sum((flux-model)**2/fluxerr**2)
    # this is ~6x faster than sklearn, LinearRegression

    print('chisq: ',chisq)
    
    # Get uncertainties
    # https://en.wikipedia.org/wiki/Weighted_least_squares
    # Parameter errors and correlation
    xcov = np.linalg.inv(A.T @ (A*weight.reshape(-1,1)))
    xerr = np.sqrt(np.diag(xcov))

    # the sklearn code uses the sqrt() of the weights
    # it uses scipy.linalg.lstsq() under the hood
    
    # Background
    if doback:
        spec = x[:-1]
        specerr = xerr[:-1]
        back = x[-1]
        backerr = xerr[-1]
        return spec,specerr,back,backerr        
    else:
        spec = x
        specerr = xerr
        return spec,specerr

def extractcol(flux,fluxerr,xcol,tr,doback=False,method='lstsq'):
    """
    This extracts flux of multiple spectra from a single column using empirical PSFs.

    Extract spectrum under the assumption that a given pixel only contributes
    to two neighboring traces, leading to a tridiagonal matrix inversion.

    Parameters
    ----------
    flux : numpy array
       The 1D flux column.
    fluxerr : numpy array
       The 1D uncertainty column.
    xcol : int
       The X column value.
    tr : Traces object
       Traces object.
    doback : boolean, optional
       Subtract the background.  False by default.
    method : str, optional
       The method to use: 'tridiag', 'lstsq'.
         triag: fit tridiagonal matrix
         lstsq: full least-squares fitting
       Default is 'lstsq'.

    Returns
    -------
    outstr : dict
        The 1D output structure with FLUX, ERR, MASK, and BACK.
    model : numpy array
        The model 2D image

    Example
    -------

    outstr,model = extractcol(flux,fluxerr,xcol,tr)

    By J. Holtzman  2011
    Translated to Python  D.Nidever May 2011
    """
    
    var = np.copy(fluxerr**2)
    
    # Only keep traces that overlap this column
    index = []
    ysize = 0
    for i in range(len(tr)):
        t = tr[i]
        if t.xmin<=xcol and t.xmax>=xcol:
            index.append(i)
            ymid = np.polyval(t.coef,xcol)
            sig = np.polyval(t.sigcoef,xcol)
            ylo = int(np.floor(ymid-5*sig))
            yhi = int(np.ceil(ymid+5*sig))
            ny = yhi-ylo+1
            ysize = np.maximum(ysize,ny)
    # Now get the PSF information
    ylo = np.zeros(len(index),int)
    yhi = np.zeros(len(index),int)
    if method=='lstsq':
        psf = np.zeros((len(index),len(flux)),float)
    else:
        psf = np.zeros((len(index),ysize),float)        
    for i in range(len(index)):
        t = tr[index[i]]
        ymid = np.polyval(t.coef,xcol)
        sig = np.polyval(t.sigcoef,xcol)
        ylo1 = int(np.floor(ymid-5*sig))
        yhi1 = int(np.ceil(ymid+5*sig))
        psf1 = t.model(xr=[xcol,xcol],yr=[ylo1,yhi1])
        psf1 = psf1[:,0]
        ylo[i] = ylo1
        yhi[i] = yhi1
        if method=='lstsq':
            psf[i,ylo1:yhi1] = psf1
        else:
            psf[i,:len(psf1)] = psf1            
        
    ntrace = len(index)

    # Solve for the fluxes
    if method=='lstsq':
        # mask??
        spec,specerr,back,backerr = _solvecol(flux,fluxerr,psf,doback=doback)
    else:
        spec,specerr,mask,back = _extractcol(flux,fluxerr,ylo,yhi,psf,doback=doback)
    
    # Initialize output arrays
    dt = [('flux',float),('err',float),('mask',int),('back',float),('trace',int)]
    out = np.zeros(ntrace,dtype=np.dtype(dt))
    out['flux'] = spec
    out['err'] = specerr
    out['mask'] = mask
    out['back'] = back
    out['trace'] = np.array(index)+1
    
    # Create the Model 2D image
    model = psfmodelcol(ylo,yhi,psf,spec,back)
    
    return out,model

@njit()
def extract_pmul(ylo,yhi,psf,n,m):
    """ Multiply two traces where they overlap. Helper function for _extractcol()."""
    # n - trace 1 index
    # m - trace 2 index
    ylo1 = ylo[n]
    yhi1 = yhi[n]
    ylo2 = ylo[m]
    yhi2 = yhi[m]
    lo = np.max(np.array([ylo1,ylo2]))
    k1 = lo-ylo1
    l1 = lo-ylo2
    hi = np.min(np.array([yhi1,yhi2]))
    k2 = hi-ylo1
    l2 = hi-ylo2
    # No overlap
    if l1<0 or l2<0 or k1<0 or k2<0:
        return 0.0
    psf1 = psf[n,:yhi[n]-ylo[n]]
    psf2 = psf[m,:yhi[m]-ylo[m]]
    if lo==hi:
        out = psf1[k1:k2]*psf2[l1:l2]
        out = out[0]
    else:
        out = np.nansum(psf1[k1:k2]*psf2[l1:l2])
    return out

@njit
def solvefibers(x,xvar,ngood,v,b,c,vvar):
    for j in np.flip(np.arange(0,ngood-1)):
        x[j] = (v[j]-c[j]*x[j+1])/b[j]
        xvar[j] = (vvar[j]+c[j]**2*xvar[j+1])/b[j]**2            
    return x,xvar

@njit
def _extractcol(flux,fluxerr,ylo,yhi,psf,doback=False):
    """
    This extracts flux of multiple spectra from a single column using empirical PSFs.

    Extract spectrum under the assumption that a given pixel only contributes
    to two neighboring traces, leading to a tridiagonal matrix inversion.

    Parameters
    ----------
    flux : numpy array
       The 1D flux column.
    fluxerr : numpy array
       The 1D uncertainty column.
    ylo : numpy array
       Array of Y start indexes.
    yhi : numpy array
       Array of Y stop indexes.
    psf : numpy array
       2-D array of PSF arrays for each trace.
    doback : boolean, optional
       Subtract the background.  False by default.

    Returns
    -------
    spec : numpyarray
        The extracted total flux value per trace.
    specerr : numpy array
        Uncertainty in SPEC.
    mask : numpy array
        Mask for SPEC.
    back : numpy array
        The background value.

    Example
    -------

    spec,specerr,mask,back = _extractcol(flux,fluxerr,ylo,yhi,psf)

    By J. Holtzman  2011
    Translated to Python  D.Nidever May 2011
    """

    nflux = len(flux)
    var = np.copy(fluxerr**2)
    ntrace = len(ylo)

    spec = np.zeros(ntrace,float)
    specerr = np.zeros(ntrace,float)
    mask = np.zeros(ntrace)
    
    # Calculate extraction matrix
    if doback:
        nback = 1 
    else:
        nback = 0
    back = 0.0
    beta = np.zeros((ntrace+nback),float)
    betavar = np.zeros((ntrace+nback),float)
    psftot = np.zeros((ntrace+nback),float)
    tridiag = np.zeros((3,ntrace+nback),float)

    # loop over all traces and load least squares matrices
    #   beta[k] = sum_i (y_i * PSF_k)
    #   alpha[k,l] = sum_i (PSF_k * PSF_l)  but stored as 3 vectors for tridiagonal case
    for k in np.arange(0,ntrace+nback):
        # Fibers
        if k <= ntrace-1:
            # Get PSF and set bad pixels to NaN
            ylo1 = ylo[k]
            yhi1 = yhi[k]
            if yhi1>nflux:
                yhi1 = nflux
            psf1 = np.copy(psf[k,:yhi1-ylo1])
            # Mask bad pixels
            bad = (~np.isfinite(flux[ylo1:yhi1+1]) | (flux[ylo1:yhi1+1] == 0))
            nbad = np.sum(bad)
            if nbad > 0:
                psf[bad] = np.nan
            psftot[k] = np.nansum(psf1)
            beta[k] = np.nansum(flux[ylo1:yhi1]*psf1)
            betavar[k] = np.nansum(var[ylo1:yhi1]*psf1**2)
        # Background
        else:
            beta[k] = np.nansum(flux[yhi1:])
            betavar[k] = np.nansum(var[yhi1:])
            psftot[k] = 1.
            
        # First fiber (on the bottom edge)
        if k==0:
            ll = 1
            for l in np.arange(k,k+2):
                tridiag[ll,k] = extract_pmul(ylo,yhi,psf,k,l)
                ll = ll + 1
        # Last fiber (on top edge)
        elif k == ntrace-1:
            ll = 0
            for l in np.arange(k-1,k+1):
                tridiag[ll,k] = extract_pmul(ylo,yhi,psf,k,l)
                ll = ll + 1
        # Background terms
        elif k > ntrace-1:
            tridiag[1,k] = len(flux[yhi1:])
        # Middle fibers (not first or last)
        else:
            ll = 0
            for l in np.arange(k-1,k+2):
                tridiag[ll,k] = extract_pmul(ylo,yhi,psf,k,l)
                ll = ll + 1

    # Good fibers
    good, = np.where(psftot > 0.1)
    ngood = len(good)
    bad, = np.where(psftot <= 0.1)
    nbad = len(bad)
    if nbad > 0:
        bad0, = np.where(bad>0)
        nbad0 = len(bad0)
        if nbad0 > 0:
            temp = tridiag[2,:]
            temp[bad[bad0]-1] = 0
            tridiag[2,:] = temp
        bad1, = np.where(bad < ntrace-1)
        nbad1 = len(bad1)
        if nbad1 > 0:
            temp = tridiag[0,:]
            temp[bad[bad1]+1] = 0
            tridiag[0,:] = temp
    if ngood>0:
        # Solving tridiagonal matrix
        # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        a = tridiag[0,:][good]
        b = tridiag[1,:][good]
        c = tridiag[2,:][good]
        v = beta[good]
        vvar = betavar[good]
        m = a[1:ngood]/b[:ngood-1]
        b[1:] = b[1:]-m*c[:ngood-1]
        v[1:] = v[1:]-m*v[:ngood-1]
        vvar[1:] = vvar[1:]+m**2*vvar[:ngood-1]
        x = np.zeros(ngood,float)
        xvar = np.zeros(ngood,float)
        x[ngood-1] = v[ngood-1]/b[ngood-1]
        xvar[ngood-1] = vvar[ngood-1]/b[ngood-1]**2
        x,xvar = solvefibers(x,xvar,ngood,v,b,c,vvar)
        
        if doback:
            spec[good[:-1]] = x[:-1]    # last one is the background
            specerr[good[:-1]] = np.sqrt(xvar[:-1])
            # mask the bad pixels
            mask[good[:-1]] = 0
            if nbad > 0:
                mask[bad] = 1
        else:
            spec[good] = x
            specerr[good] = np.sqrt(xvar)
            # mask the bad pixels
            mask[good] = 0
            if nbad > 0:
                mask[bad] = 1
                
    # No good fibers for this column
    else:
        spec[:] = 0.0
        specerr[:] = 1e30
        mask[:] = 1

    if doback:
        back = x[ngood-1]
    else:
        back = 0.0
            
    # Catch any NaNs (shouldn't be there, but ....)
    bad = ~np.isfinite(spec)
    nbad = np.sum(bad)
    if nbad > 0:
        spec[bad] = 0.
        specerr[bad] = 1e30
        mask[bad] = 1
    
    return spec,specerr,mask,back
