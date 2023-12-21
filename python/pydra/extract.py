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
from numba import njit
from . import utils

import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import plotting as pl


# Ignore these warnings
import warnings
warnings.filterwarnings("ignore", message="OptimizeWarning: Covariance of the parameters could not be estimated")

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
        


def extract_pmul(p1lo,p1hi,img,p2):
    """ Helper function for extract()."""
    
    lo = np.max([p1lo,p2['lo']])
    k1 = lo-p1lo
    l1 = lo-p2['lo']
    hi = np.min([p1hi,p2['hi']])
    k2 = hi-p1lo
    l2 = hi-p2['lo']
    # No overlap
    if l1<0 or l2<0 or k1<0 or k2<0:
        out = np.zeros(2048,float)
        return out
    if lo>hi:
        out = np.zeros(2048,float)
    img2 = p2['img'].T  # transpose
    if lo==hi:
        out = img[:,k1:k2+1]*img2[:,l1:l2+1]
    else:
        out = np.nansum(img[:,k1:k2+1]*img2[:,l1:l2+1],axis=1)
    if out.ndim==2:
        out = out.flatten()   # make sure it's 1D
    return out

@njit
def solvefibers(x,xvar,ngood,v,b,c,vvar):
    for j in np.flip(np.arange(0,ngood-1)):
        x[j] = (v[j]-c[j]*x[j+1])/b[j]
        xvar[j] = (vvar[j]+c[j]**2*xvar[j+1])/b[j]**2            
    return x,xvar

def epsfmodel(epsf,spec,skip=False,subonly=False,fibers=None,yrange=[0,2048]):
    """ Create model image using EPSF and best-fit values."""
    # spec [2048,300], best-fit flux values
    
    ntrace = len(epsf)
    if fibers is None:
        fibers = np.arange(ntrace)
    
    # Create the Model 2D image
    if yrange is not None:
        model = np.zeros((2048,yrange[1]-yrange[0]),float)
        ylo = yrange[0]
    else:
        ylo = 0
        model = np.zeros((2048,2048),float)
    t = np.copy(spec)
    bad = (t<=0)
    if np.sum(bad)>0:
        t[bad] = 0
    for k in fibers:
        nf = 1
        ns = 0
        if subonly:
            junk, = np.where(subonly==k)
            nf = len(junk)
        if skip:
            junk, = np.where(skip==k)
            ns = len(junk)
        if nf > 0 and ns==0:
            p1 = epsf[k]
            lo = epsf[k]['lo']
            hi = epsf[k]['hi']
            img = p1['img'].T
            rows = np.ones(hi-lo+1,int)
            fiber = epsf[k]['fiber']
            model[:,lo-ylo:hi+1-ylo] += img[:,:]*(rows.reshape(-1,1)*t[:,fiber]).T                                    
    model = model.T

    return model


def extractcol(flux,fluxerr,epsf,doback=False,skip=False,subonly=False):
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
    epsf : list
       A list with the empirical PSF.
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

    outstr,back,model = extractcol(image,epsf)

    By J. Holtzman  2011
      Incorporated into ap2dproc.pro  D.Nidever May 2011
    """
    
    ntrace = len(epsf)
    fibers = np.array([e['fiber'] for e in epsf])
    var = np.copy(fluxerr**2)
        
    # Initialize output arrays
    dt = [('flux',float),('err',float),('mask',int),('back',float),('fiber',int)]
    out = np.zeros(ntrace,dtype=np.dtype(dt))
    out['fiber'] = fibers
    
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

    for k in np.arange(0,ntrace+nback):
        # Fibers
        if k <= ntrace-1:
            # Get EPSF and set bad pixels to NaN
            p1 = epsf[k]
            lo = epsf[k]['lo']
            hi = epsf[k]['hi']
            bad = (~np.isfinite(flux[lo:hi+1]) | (flux[lo:hi+1] == 0))
            nbad = np.sum(bad)
            img = np.copy(p1['img'].T)   # transpose
            if nbad > 0:
                img[bad] = np.nan
            psftot[k] = np.nansum(img,axis=1)
            beta[k] = np.nansum(flux[lo:hi+1]*img,axis=1)
            betavar[k] = np.nansum(var[lo:hi+1]*img**2,axis=1)
        # Background
        else:
            beta[k] = np.nansum(flux[lo:hi+1],axis=1)
            betavar[k] = np.nansum(var[lo:hi+1],axis=1)
            psftot[k] = 1.
            
        # First fiber (on the bottom edge)
        if k==0:
            ll = 1
            for l in np.arange(k,k+2):
                tridiag[ll,k] = extract_pmul(p1['lo'],p1['hi'],img,epsf[l])
                ll += 1
        # Last fiber (on top edge)
        elif k == ntrace-1:
            ll = 0
            for l in np.arange(k-1,k+1):
                tridiag[ll,k] = extract_pmul(p1['lo'],p1['hi'],img,epsf[l])
                ll += 1
        # Background terms
        elif k > ntrace-1:
            tridiag[1,k] = hi-lo+1
        # Middle fibers (not first or last)
        else:
            ll = 0
            for l in np.arange(k-1,k+2):
                tridiag[ll,k] = extract_pmul(p1['lo'],p1['hi'],img,epsf[l])
                ll += 1

    # Good fibers
    good, = np.where(psftot > 0.5)
    ngood = len(good)
    bad, = np.where(psftot <= 0.5)
    nbad = len(bad)
    if nbad > 0:
        bad0, = np.where(bad>0)
        nbad0 = len(bad0)
        if nbad0 > 0:
            tridiag[2,bad[bad0]-1]=0 
        bad1, = np.where(bad < ntrace-1)
        nbad1 = len(bad1)
        if nbad1 > 0:
            tridiag[0,bad[bad1]+1] = 0 
    if ngood>0:
        a = tridiag[0,good]
        b = tridiag[1,good]
        c = tridiag[2,good]
        v = beta[good]
        vvar = betavar[good]
        m = a[1:ngood]/b[0:ngood-1]
        b[1:] = b[1:]-m*c[0:ngood-1]
        v[1:] = v[1:]-m*v[0:ngood-1]
        vvar[1:] = vvar[1:]+m**2*vvar[0:ngood-1]
        x = np.zeros(ngood,float)
        xvar = np.zeros(ngood,float)
        x[ngood-1] = v[ngood-1]/b[ngood-1]
        xvar[ngood-1] = vvar[ngood-1]/b[ngood-1]**2
        # Use numba to speed up this slow loop
        #for j in np.flip(np.arange(0,ngood-1)):
        #    x[j] = (v[j]-c[j]*x[j+1])/b[j]
        #    xvar[j] = (vvar[j]+c[j]**2*xvar[j+1])/b[j]**2
        x,xvar = solvefibers(x,xvar,ngood,v,b,c,vvar)
        out['flux'][good] = x
        out['err'][good] = np.sqrt(xvar)
        # mask the bad pixels
        out['mask'][good] = 0
        if nbad > 0:
            out['mask'][bad] = 1
            
    # No good fibers for this column
    else:
        out['flux'][:] = 0
        out['err'][:] = 1e30
        out['mask'][:] = 1

    if doback:
        back = x[ngood-1]
        out['back'] = back

    import pdb; pdb.set_trace()
            
    # Catch any NaNs (shouldn't be there, but ....)
    bad = ~np.isfinite(out['flux'])
    nbad = np.sum(bad)
    if nbad > 0:
        out['flux'][bad] = 0.
        out['err'][bad] = 1e30
        out['mask'][bad] = 1

    # Create the Model 2D image
    model = epsfmodel(epsf,out['flux'],subonly=subonly,skip=skip)

    
    return out,model
