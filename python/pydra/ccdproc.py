#!/usr/bin/env python

"""CCDPROC.PY - Generic CCD image calibration/reduction

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20211021'  # yyyymmdd    

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
from glob import glob
import time
import re
import subprocess
import warnings
from astropy.modeling.models import Gaussian2D
from scipy.interpolate import RectBivariateSpline
from dlnpyutils import utils as dln
#import matplotlib
#matplotlib.use('nbagg')

def datadir():
    """ Get package data directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

def mad(data,axis=None):
    """ Calculate the median absolute deviation."""
    data = np.asanyarray(data)
    data_median = np.median(data, axis=axis)
    result = np.median(np.abs(data-data_median), axis=axis, overwrite_input=True)
    if axis is None:
    # return scalar version
        result = result.item()
    return result * 1.482602218505602
    
def ccdlist(input=None):
    if input is None: input='*.fits'
    if type(input) is list:
        files = input
    else:
        files = glob(input)
    nfiles = len(files)
    dt = np.dtype([('file',str,100),('object',str,100),('naxis1',int),('naxis2',int),
                   ('imagetyp',str,100),('exptime',float),('filter',str,100),
                   ('dateobs',str,50),('jd',float)])
    cat = np.zeros(nfiles,dtype=dt)
    for i,f in enumerate(files):
        base = os.path.basename(f)
        base = base.split('.')[0]
        h = fits.getheader(f)
        cat['file'][i] = f
        cat['object'][i] = h.get('object')
        cat['naxis1'][i] = h.get('naxis1')
        cat['naxis2'][i] = h.get('naxis2')
        cat['imagetyp'][i] = h.get('imagetyp')
        cat['exptime'][i] = h.get('exptime')
        cat['filter'][i] = h.get('filter')
        cat['dateobs'][i] = h.get('date-obs')
        cat['jd'][i] = h.get('jd')        
        print(base+'  '+str(cat['naxis1'][i])+'  '+str(cat['naxis2'][i])+'  '+cat['imagetyp'][i]+'  '+str(cat['exptime'][i])+'  '+cat['filter'][i])
    return cat

def library():
    """ Get calibration library file info."""
    ddir = datadir()
    files = []
    for p in ['*.fit','*.fit.gz','*.fits','*.fits.gz']:
        files += glob(ddir+p)
    nfiles = len(files)
    if nfiles==0:
        return None
    cat = np.zeros(nfiles,dtype=np.dtype([('file',str,200),('name',str,100),('naxis1',int),
                                          ('naxis2',int),('imagetyp',str,50),
                                          ('type',str,50),('filter',str,40),('dateobs',str,30),
                                          ('jd',float),('master',bool)]))
    for i in range(nfiles):
        head = fits.getheader(files[i])
        cat['file'][i] = files[i]
        cat['name'][i] = os.path.basename(files[i])
        cat['naxis1'][i] = head.get('naxis1')
        cat['naxis2'][i] = head.get('naxis2')        
        imagetyp = head.get('imagetyp')
        cat['imagetyp'][i] = imagetyp
        for t in ['bias','dark','flat','light']:
            if t in imagetyp.lower():
                cat['type'][i] = t
        if 'bpm' in cat['name'][i]:
            cat['type'][i] = 'bpm'
        cat['filter'][i] = head.get('filter')    
        cat['dateobs'][i] = head.get('date-obs')
        cat['jd'][i] = head.get('jd')
        if 'master' in cat['name'][i]:
            cat['master'][i] = True
        else:
            cat['master'][i] = False
    return cat

def fixheader(head):
    """ Update the headers."""
    head2 = head.copy()
    nx = head['NAXIS1']
    ny = head['NAXIS2']
    # SBIG camera with overscan: [3468,2728]
    if (nx==3468) and (ny==2728):
        head2['BIASSEC1'] = '[1:41,1:2728]'
        head2['BIASSEC2'] = '[3430:3465,1:2728]'
        head2['TRIMSEC'] = '[42:3429,15:2726]'
        head2['DATASEC'] = '[42:3429,15:2726]'
        head2['RDNOISE'] = (4.5, 'readnoise in e-')
        head2['GAIN'] = (0.15759900212287903, 'Electronic gain in e-/ADU')
        head2['BUNIT'] = 'ADU'        
    # SBIG camera without overscan: [3380,2704]        
    elif (nx==3380) and (ny==2704):
        head2['BIASSEC'] = '[1:3380,1:2]'
        head2['TRIMSEC'] = '[1:3380,3:2704]'
        head2['DATASEC'] = '[1:3380,3:2704]'
        head2['RDNOISE'] = (4.5, 'readnoise in e-')
        head2['GAIN'] = (0.15759900212287903, 'Electronic gain in e-/ADU')
        head2['BUNIT'] = 'ADU'        
    # LhiresIII and Atik camera [2749,2199]
    elif (nx==2749) and (ny==2199):
        head2['BIASSEC'] = '[0:0,0:0]'
        head2['TRIMSEC'] = '[1:2749,1:2199]'
        head2['DATASEC'] = '[1:2749,1:2199]'
        head2['RDNOISE'] = (5.0, 'readnoise in e-')
        head2['GAIN'] = (0.27, 'Electronic gain in e-/ADU')
        head2['BUNIT'] = 'ADU'        
    else:
        warnings.warn('Image size not understood')
    return head2

    
def overscan(im,head,verbose=False):
    """ This calculate the overscan and subtracts it from the data and then trims off the overscan region"""
    # y = [0:40] and [3429:3464]
    # x = [0:13] and [2726:2727]
    # DATA = [14:2725,41:3428]
    # 2712 x 3388
    nx,ny = im.shape

    # Use trimsec
    trimsec = head.get('TRIMSEC')
    if trimsec is None:
        warnings.warn('No TRIMSEC found in header')
        return im,head
    trim = [int(s) for s in re.findall(r'\d+',trimsec)]

    # biassec
    biassec = head.get('BIASSEC')
    biassec2 = None
    if biassec is None:
        biassec = head.get('BIASSEC1')
        biassec2 = head.get('BIASSEC2')        
    if biassec is None:
        raise ValueError('No BIASSEC found in header')
    bias = [int(s) for s in re.findall(r'\d+',biassec)]    

    # No overscan region, [0:0, 0:0]
    if bias[0]==bias[1] and bias[2]==bias[3]:
        print('No overscan region')
        return im,head
    
    # Y first, then X
    o = im[bias[2]-1:bias[3],bias[0]-1:bias[1]]
    # check for second biassec
    if biassec2 is not None:
        bias2 = [int(s) for s in re.findall(r'\d+',biassec2)]
        o2 = im[bias2[2]-1:bias2[3],bias2[0]-1:bias2[1]]
        o = np.hstack((o,o2))
        
    # Subtract overscan
    oshape = o.shape
    if oshape[0] > oshape[1]:
        # Take the mean        
        mno = np.mean(o,axis=1)
        # Fit line to it
        coef = np.polyfit(np.arange(nx),mno,1)
        fit = np.poly1d(coef)(np.arange(nx))
        # Subtract from entire image
        oim = np.repeat(fit,ny).reshape(nx,ny)
        out = im.astype(float)-oim
    else:
        # Take the mean        
        mno = np.mean(o,axis=0)
        # Fit line to it
        coef = np.polyfit(np.arange(ny),mno,1)
        fit = np.poly1d(coef)(np.arange(ny))
        # Subtract from entire image
        oim = np.repeat(fit,nx).reshape(nx,ny)
        out = im.astype(float)-oim        
        
    # Trim the overscan
    out = out[trim[2]-1:trim[3],trim[0]-1:trim[1]]
    #out = out[14:2726,41:3429]
    # Update header
    nx1, ny1 = out.shape
    head2 = head.copy()
    head2['NAXIS1'] = ny1
    head2['NAXIS2'] = nx1
    head2['BITPIX'] = -32
    if biassec2 is not None:
        head2['BIASSEC1'] = biassec
        head2['BIASSEC2'] = biassec2
    else:
        head2['BIASSEC'] = biassec        
    head2['TRIMSEC'] = trimsec
    head2['OVSNMEAN'] = np.mean(oim)
    head2['TRIM'] = time.ctime()+' Trim is '+trimsec
    if biassec2 is not None:
        head2['OVERSCAN'] = time.ctime()+' Overscan is '+biassec+' and '+biassec2+', mean '+str(np.mean(oim))
    else:
        head2['OVERSCAN'] = time.ctime()+' Overscan is '+biassec+', mean '+str(np.mean(oim))
    if verbose:
        print(head2['OVERSCAN'])
        print(time.ctime()+' Trimming to '+trimsec)
    return out, head2

def fixpixconv(im,mask,yind,xind,filt):
    ny,nx = im.shape
    npix,npix = filt.shape
    nh = npix//2
    # image indices
    ylo = np.maximum(yind-nh,0)
    yhi = np.minimum(yind+nh+1,ny)
    xlo = np.maximum(xind-nh,0)
    xhi = np.minimum(xind+nh+1,nx)
    slc = (slice(ylo,yhi,None),slice(xlo,xhi,None))
    # filter indices
    fylo = np.maximum(yind-nh,0)-yind+nh
    fyhi = np.minimum(yind+nh+1,ny)-yind+nh
    fxlo = np.maximum(xind-nh,0)-xind+nh
    fxhi = np.minimum(xind+nh+1,nx)-xind+nh
    fslc = (slice(fylo,fyhi,None),slice(fxlo,fxhi,None))
    ngd = np.sum(mask[slc])
    if ngd==0:
        return None
    totf = np.sum(filt[fslc])
    new = np.sum(im[slc]*(mask[slc]==0)*filt[fslc])/totf
    return new
    
def fixpix(im,mask,head,verbose=False):
    """ Interpolate over bad pixels."""
    ny,nx = im.shape
    
    yind,xind = np.where(mask>0)
    nind = len(xind)
    # Make filters
    xx,yy = np.meshgrid(np.arange(3),np.arange(3))
    filt3 = np.exp(-0.5*((xx-1)**2+(yy-1)**2)/1.0**2)
    xx,yy = np.meshgrid(np.arange(7),np.arange(7))
    filt7 = np.exp(-0.5*((xx-3)**2+(yy-3)**2)/1.0**2)
    xx,yy = np.meshgrid(np.arange(11),np.arange(11))
    filt11 = np.exp(-0.5*((xx-5)**2+(yy-5)**2)/1.0**2)
    
    # Loop over pixels
    nfix = 0
    for i in range(nind):
        yind1 = yind[i]
        xind1 = xind[i]
        # convolve with filter
        new = fixpixconv(im,mask,yind1,xind1,filt3)
        if new is None:
            new = fixpixconv(im,mask,yind1,xind1,filt7)
        if new is None:
            new = fixpixconv(im,mask,yind1,xind1,filt11)
        if new is not None:
            im[yind1,xind1] = new
            mask[yind1,xind1] += 4
            nfix += 1
            
    head['FIXPIX'] = time.ctime()+' FIXPIX: '+str(nfix)+' pixels interpolated over'
    if verbose:
        print(time.ctime()+' Fixpix: '+str(nfix)+' pixels interpolated over')
    
    return im,mask,head

def masterbias(files,med=False,outfile=None,clobber=True,verbose=False):
    """
    Load the bias images.  Overscan correct and trim them.  Then average them.

    Parameters
    ----------
    files : list
        List of bias FITS files.
    med : boolean, optional
        Use the median of all the files.  By default med=False and the mean is calculated.
    outfile : string, optional
        Filename to write the master bias image to.
    clobber : boolean, optional
        If the output file already exists, then overwrite it.  Default is True.
    verbose : boolean, optional
        Verbose output to the screen.  Default is False.

    Returns
    -------
    aim : numpy image
        The 2D master bias image.
    ahead : header dictionary
        The master bias header.

    Example
    -------

    bias, bhead = masterbias(bias_files)

    """

    nfiles = len(files)
    if verbose:
        print('Creating master bias using '+str(nfiles)+' files')
    # File loop    
    for i in range(nfiles):
        im,head = fits.getdata(files[i],0,header=True)
        sh = im.shape
        if verbose:
            print(str(i+1)+' '+files[i]+' ['+str(sh[1])+','+str(sh[0])+']')        
        # Fix header, if necessary
        if (head.get('TRIMSEC') is None) | (head.get('BIASSEC') is None):
            head = fixheader(head)
        # Check image type
        imagetyp = head.get('IMAGETYP')
        exptime = head.get('EXPTIME')
        if imagetyp is not None:
            if 'bias' not in imagetyp.lower() and 'zero' not in imagetyp.lower() and exptime != 0.0:
                raise ValueError(files[i]+' is not a bias')
        # Image processing, overscan+trim
        im2,head2 = ccdproc(im,head)
        # Initialize array
        if i==0:
            ny,nx = im2.shape
            if med:
                imarr = np.zeros((ny, nx, nfiles),float)
            else:
                totim = np.zeros(im2.shape,float)
        if med:
            imarr[:,:,i] = im2
        else:
            totim += im2
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    # Final calculation
    if med:
        aim = np.median(imarr,axis=2)
        ahead['HISTORY'] = 'Median combine'
    else:
        aim = totim/nfiles
        ahead['HISTORY'] = 'Mean combine'
    ahead['NCOMBINE'] = nfiles
    ahead['HISTORY'] = time.ctime()+' bias combine'
    aim = aim.astype(np.float32)  # convert to 32 bit    
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        if verbose:
            print('Writing master bias to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)
    
    return aim, ahead


def masterdark(files,zero,med=False,outfile=None,clobber=True,verbose=False):
    """
    Load the dark images.  Overscan correct and trim them.  zero subtract.  Then average them.

    Parameters
    ----------
    files : list
        List of dark FITS files.
    zero : numpy image or str
        Master bias.  This can be the image or the filename.
    med : boolean, optional
        Use the median of all the files.  By default med=False and the mean is calculated.
    outfile : string, optional
        Filename to write the master dark image to.
    clobber : boolean, optional
        If the output file already exists, then overwrite it.  Default is True.
    verbose : boolean, optional
        Verbose output to the screen.  Default is False.

    Returns
    -------
    aim : numpy image
        The 2D master dark image.
    ahead : header dictionary
        The master dark header.

    Example
    -------

    dark, dhead = masterdark(dark_files,zero)

    """

    nfiles = len(files)
    if verbose:
        print('Creating master dark using '+str(nfiles)+' files')

    # Load master bias if filename input
    if type(zero) is str:
        zerofile = zero
        zero,zhead = fits.getdata(zerofile,header=True)
        
    # File loop        
    for i in range(nfiles):
        im,head = fits.getdata(files[i],0,header=True)
        sh = im.shape
        if verbose:
            print(str(i+1)+' '+files[i]+' ['+str(sh[1])+','+str(sh[0])+'] '+str(head['exptime'])+' sec')
        # Fix header, if necessary
        if (head.get('TRIMSEC') is None) | (head.get('BIASSEC') is None):
            head = fixheader(head)        
        # Check image type
        imagetyp = head.get('IMAGETYP')
        if imagetyp is not None:
            if 'dark' not in imagetyp.lower():
                raise ValueError(files[i]+' is not a dark')
        # Image processing, overscan+trim+zercorr        
        im2,head2 = ccdproc(im,head,zero=zero)
        # Initialize array
        if i==0:
            ny,nx = im2.shape
            if med:
                imarr = np.zeros((ny,nx, nfiles),float)
            else:
                totim = np.zeros(im2.shape,float)
        if med:
            imarr[:,:,i] = im2 / head['exptime']  # divide by exposure time
        else:
            totim += im2 / head['exptime']
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    # Final calculation
    if med:
        aim = np.median(imarr,axis=2)
        ahead['HISTORY'] = 'Median combine'        
    else:
        aim = totim/nfiles
        ahead['HISTORY'] = 'Mean combine'
    # Make sure they are all non-negative
    aim = np.maximum(aim,0)
    ahead['NCOMBINE'] = nfiles
    ahead['HISTORY'] = time.ctime()+' dark combine'
    aim = aim.astype(np.float32)  # convert to 32 bit    
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        if verbose:
            print('Writing master dark to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)
    
    return aim, ahead

def masterflat(files,zero,dark,med=False,sigclip=False,outfile=None,clobber=True,verbose=False):
    """
    Load the flat images.  Overscan correct and trim them.  Bias and dark subtract.
    Then divide by median and average them.

    Parameters
    ----------
    files : list
        List of flat FITS files.
    zero : numpy image or str
        Master bias.  This can be the image or the filename.
    dark : numpy image or str
        Master dark.  This can be the image or the filename.
    med : boolean, optional
        Use the median of all the files.  By default med=False and the mean is calculated.
    sigclip : boolean, optional
        Use sigma clipped mean of all the files.  By default sigclip=False and the
          standard mean is calculated.
    outfile : string, optional
        Filename to write the master flat image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.
        If the output file already exists, then overwrite it.  Default is True.
    verbose : boolean, optional
        Verbose output to the screen.  Default is False.

    Returns
    -------
    aim : numpy image
        The 2D master flat image.
    ahead : header dictionary
        The master flat header.

    Example
    -------

    flat, fhead = masterflat(flat_files,zero,dark)

    """

    nfiles = len(files)
    if verbose:
        print('Creating master flat using '+str(nfiles)+' files')

    # Load master bias if filename input
    if type(zero) is str:
        zerofile = zero
        zero,zhead = fits.getdata(zerofile,header=True)
    # Load master dark if filename input
    if type(dark) is str:
        darkfile = dark
        dark,dhead = fits.getdata(darkfile,header=True)        

    # File loop
    for i in range(nfiles):
        im,head = fits.getdata(files[i],0,header=True)
        sh = im.shape
        # Fix header, if necessary
        if (head.get('TRIMSEC') is None) | (head.get('BIASSEC') is None):
            head = fixheader(head)        
        # Check image type
        imagetyp = head.get('IMAGETYP')
        if imagetyp is not None:
            if 'flat' not in imagetyp.lower():
                raise ValueError(files[i]+' is not a flat')
        # Image processing, overscan+trim+zercorr+darkcorr
        im2,head2 = ccdproc(im,head,zero=zero,dark=dark)
        if verbose:
            print(str(i+1)+' '+files[i]+' ['+str(sh[1])+','+str(sh[0])+'] '+str(head['exptime'])+' sec  %8.2f ADU' %(np.median(im2)))
        # Initialize array
        if i==0:
            ny,nx = im2.shape
            if med or sigclip:
                imarr = np.zeros((ny,nx,nfiles),float)
            else:
                totim = np.zeros(im2.shape,float)
        if med or sigclip:
            imarr[:,:,i] = im2 / np.median(im2)   # divide by median flux
        else:
            totim += im2 / np.median(im2)
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    # Final calculation
    if med:
        aim = np.median(imarr,axis=2)
        ahead['HISTORY'] = 'Median combine'
    elif sigclip:
        mim = np.median(imarr,axis=2)
        sig = mad(imarr)
        diff = imarr - mim.reshape(mim.shape+(-1,))
        mask = np.abs(diff) > 3*sig
        #mask = np.abs(imarr-1.0) > 3*sig        
        temp = imarr.copy()
        temp[mask] = np.nan
        aim = np.nanmean(temp,axis=2)
        ahead['HISTORY'] = 'Sigma clipped mean combine'
    else:
        aim = totim / nfiles
        ahead['HISTORY'] = 'Mean combine'
    ahead['NCOMBINE'] = nfiles
    ahead['HISTORY'] = time.ctime()+' flat combine'
    aim = aim.astype(np.float32)  # convert to 32 bit
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        if verbose:
            print('Writing master flat to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)

    return aim, ahead

def makebpm(zero,dark=None,flat=None,maxzero=100,maxdark=1,maxflat=1.2,
            outfile=None,verbose=False,clobber=True,compress=False):
    """
    Create bad pixel mask from master bias, dark and flat.

    Parameters
    ----------
    zero : filename or numpy 2D array, optional
        The master bias.  Either the 2D image or the filename.
    dark : filename or numpy 2D array, optional
        The master dark.  Either the 2D image or the filename.
    flat : filename or numpy 2D array, optional
        The master flat.  Either the 2D image or the filename.
    maxzero : int, optional
        The cutoff to use for the bias image.  Values above that will
          be considered bad.  Default is 100.
    maxdark : int, optional
        The cutoff to use for the dark image.  Values above that will
          be considered bad.  Default is 1.0
    maxflat : int, optional
        The cutoff to use for the flat image.  Values above that will
          be considered bad.  Default is 1.2
    outfile : string, optional
        Filename to write the processed image to.
    verbose : boolean, optional
        Verbose output to the screen.
    clobber : boolean, optional
        If the output file already exists, then overwrite it.
    compress : boolean, optional
        Gzip compress output file.  Default is False.

    Returns
    -------
    bpm : numpy image
        The 2D processed image.
    fhead : header dictionary
        The header for the processed image.

    Example
    -------

    bpm, head = makebpm(zero,dark,flat)

    """  

    # Load master bias if filename input
    if type(zero) is str:
        zerofile = zero
        zero,zhead = fits.getdata(zerofile,header=True)
    # Load master dark if filename input
    if type(dark) is str:
        darkfile = dark
        dark,dhead = fits.getdata(darkfile,header=True)
    # Load master flat if filename input
    if type(flat) is str:
        flatfile = flat
        flat,fhead = fits.getdata(flatfile,header=True)                

    bhead = zhead.copy()

    # Initialize array
    ny,nx = zero.shape
    bpm = np.zeros((ny,nx),int)
    
    # Check zero
    bpm[zero>maxzero] = 1
    # Check dark
    if dark is not None:
        bpm[dark>maxdark] = 1
    # Check flat
    if flat is not None:
        bpm[flat>maxflat] = 1     

    nbad = np.sum(bpm)
    bhead['HISTORY'] = str(nbad)+' pixels marked as bad'
    bhead['HISTORY'] = time.ctime()+' BPM creation'
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        if verbose:
            print('Writing bpm to '+outfile)
        hdu = fits.PrimaryHDU(bpm,bhead).writeto(outfile)
    
    return bpm, bhead


def masterspecflat(files,zero=None,dark=None,spectral_axis=1,outfile=None,
                 med=False,sigclip=False,clobber=True,verbose=False):
    """

    Make master spectrum flat.  Remove spectral and spatial variations.

    Parameters
    ----------
    files : list
        List of flat FITS files.
    zero : numpy image or str
        Master bias.  This can be the image or the filename.
    dark : numpy image or str
        Master dark.  This can be the image or the filename.
    spectral_axis : int
        The spectral axis.  Default is 1, the x-axis.
    med : boolean, optional
        Use the median of all the files.  By default med=False and the mean is calculated.
    sigclip : boolean, optional
        Use sigma clipped mean of all the files.  By default sigclip=False and the
          standard mean is calculated.
    outfile : string, optional
        Filename to write the master flat image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.
        If the output file already exists, then overwrite it.  Default is True.
    verbose : boolean, optional
        Verbose output to the screen.  Default is False.

    Returns
    -------
    flat : numpy image
        The 2D flat image.
    fhead : header dictionary
        The header for the flat image.

    Example
    -------

    flat, fhead = masterspecflat(files,zero,dark)

    """  

    
    # Remove variations in both the spectral and spatial dimensions
    
    # Spatial axis
    if spectral_axis==0:
        spatial_axis = 1
    else:
        spatial_axis = 0

    nfiles = len(files)
    if verbose:
        print('Creating master spectrum flat using '+str(nfiles)+' files')

    # Load master bias if filename input
    if type(zero) is str:
        zerofile = zero
        zero,zhead = fits.getdata(zerofile,header=True)
    # Load master dark if filename input
    if type(dark) is str:
        darkfile = dark
        dark,dhead = fits.getdata(darkfile,header=True)        

    # File loop
    for i in range(nfiles):
        im,head = fits.getdata(files[i],0,header=True)
        sh = im.shape
        # Fix header, if necessary
        if (head.get('TRIMSEC') is None) | (head.get('BIASSEC') is None):
            head = fixheader(head)        
        # Check image type
        imagetyp = head.get('IMAGETYP')
        if imagetyp is not None:
            if 'flat' not in imagetyp.lower():
                raise ValueError(files[i]+' is not a flat')
        # Image processing, overscan+trim+zercorr+darkcorr
        im2,head2 = ccdproc(im,head,zero=zero,dark=dark)
        if verbose:
            print(str(i+1)+' '+files[i]+' ['+str(sh[1])+','+str(sh[0])+'] '+str(head['exptime'])+' sec  %8.2f ADU' %(np.median(im2)))
        # Initialize array
        if i==0:
            ny,nx = im2.shape
            if med or sigclip:
                imarr = np.zeros((ny,nx,nfiles),float)
            else:
                totim = np.zeros(im2.shape,float)

        # Median smooth the image with rebinning and interplation
        nbin = 25
        medim = dln.rebin(im2,binsize=(nbin,nbin),med=True)
        x,y = np.meshgrid(np.arange(im2.shape[0]),np.arange(im2.shape[1]))
        xmed = dln.rebin(x,binsize=(nbin,nbin),med=True)
        ymed = dln.rebin(y,binsize=(nbin,nbin),med=True)
        fn = RectBivariateSpline(xmed[0,:],ymed[:,0],medim,kx=3,ky=3,s=0)
        sm = fn(x[0,:],y[:,0],grid=True)
        relim = im2/sm
        # Make sure that are no residual spectral features
        specmed = np.median(relim,axis=spatial_axis)
        if spectral_axis==1:
            relim /= specmed.reshape(1,-1)
        else:
            relim /= specmed.T.reshape(-1,1)  
                
        if med or sigclip:
            imarr[:,:,i] = relim   # divide by smoothed image
        else:  
            totim += relim      # divide by smoothed image
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    # Final calculation
    if med:
        aim = np.median(imarr,axis=2)
        ahead['HISTORY'] = 'Median combine'
    elif sigclip:
        mim = np.median(imarr,axis=2)
        sig = mad(imarr)
        diff = imarr - mim.reshape(mim.shape+(-1,))
        mask = np.abs(diff) > 3*sig
        #mask = np.abs(imarr-1.0) > 3*sig        
        temp = imarr.copy()
        temp[mask] = np.nan
        aim = np.nanmean(temp,axis=2)
        ahead['HISTORY'] = 'Sigma clipped mean combine'
    else:
        aim = totim / nfiles
        ahead['HISTORY'] = 'Mean combine'
    ahead['NCOMBINE'] = nfiles
    ahead['HISTORY'] = time.ctime()+' flat combine'
    aim = aim.astype(np.float32)  # convert to 32 bit
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        if verbose:
            print('Writing master spectrum flat to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)

    return aim, ahead


def ccdproc(data,head=None,bpm=None,zero=None,dark=None,flat=None,outfile=None,outsuffix='_red',
            verbose=False,clobber=True,compress=False,fix=False):
    """
    Overscan subtract, trim, subtract master zero, subtract master dark, flat field.

    Parameters
    ----------
    data : list or numpy 2D array
        This can either be (1) a list of image filenames, (2) a file with a list of
            names image filenames, or (3) a 2D image (head must also be input).
    head : header dictionary, optional
        The header if a single image is input (data).
    bpm : filename, numpy 2D array, or boolean, optional
        The bad pixel mask.  Either the 2D image, the filename, or
          a boolean.  If bpm=True, then a library BPM will be used.
    zero : filename, numpy 2D array, or boolean, optional
        The master bias.  Either the 2D image, the filename, or
          a boolean.  If zero=True, then a library zero will be used. 
    dark : filename, numpy 2D array, or boolean, optional
        The master dark.  Either the 2D image, the filename, or
          a boolean.  If dark=True, then a library dark will be used. 
    flat : filename, numpy 2D array, or boolean, optional
        The master flat.  Either the 2D image, the filename, or
          a boolean.  If flat=True, then a library flat will be used. 
    fix : boolean, optional
        Interpolate over bad pixels.  Default is False.
    outfile : string or boolean, optional
        Filename to write the processed image to.  If outfile=True, then
          the output filename will be the input filename with outsuffix
          added (i.e. image.fits -> image_red.fits).
    outsuffix : string, optional
        Suffix to use for output files. Default is "_red".
    verbose : boolean, optional
        Verbose output to the screen.
    clobber : boolean, optional
        If the output file already exists, then overwrite it.
    compress : boolean, optional
        Gzip compress output file.  Default is False.

    Returns
    -------
    fim : numpy image
        The 2D processed image.
    fhead : header dictionary
        The header for the processed image.

    Example
    -------

    flat, fhead = ccdproc(files,bpm,zero,dark,flat)

    """

    # Check the inputs
    if type(data) is str:     # filname input
        files = [data]
        # Check if it's a FITS filename or a list
        try:
            head = fits.getheader(data)
        except:
            # This might be a list of filenames
            f = open(data,'r')
            files = f.readlines()
            f.close()
            files = [l.rstrip('\n') for l in files]  # strip newlines
    elif type(data) is list:  # list of files input
        files = data
    elif type(data) is np.ndarray:  # numpy array
        if data.ndim==1 and (data.dtype.type==str):  # array of filenames
            files = list(data)
        elif data.ndim==2:  # 2D image
            files = ['']
            im = data
        else:
            raise ValueError('Input numpy data not understood')
    else:
        raise ValueError('Input data not understood')
    nfiles = len(files)
    
    # Get calibration library information
    cals = library()

    # Load calibration files
    #-----------------------
    # -- BPM ---
    if type(bpm) is str:
        if os.path.exists(bpm):
            bpmim,bpmhead = fits.getdata(bpm,0,header=True)
        else:
            raise ValueError(bpm+' NOT FOUND')
    # Use calibration library file
    elif bpm is True:
        bpmind, = np.where(cals['type']=='bpm')
        if len(bpmind)==0:
            raise ValueError('No library BPM found')
        bpmfile = cals['file'][bpmind[0]]
        if os.path.exists(bpmfile):
            bpmim,bpmhead = fits.getdata(bpmfile,0,header=True)
        else:
            raise ValueError('Library '+bpmfile+' file NOT FOUND')
    # Image input
    else:
        bpmim = bpm
    # --- ZERO ---
    # Filename input
    if type(zero) is str:
        if os.path.exists(zero):
            zeroim,zerohead = fits.getdata(zero,0,header=True)
        else:
            raise ValueError(zero+' NOT FOUND')
    # Use calibration library file
    elif zero is True:
        zeroind, = np.where((cals['type']=='bias') & (cals['master']==True))
        if len(zeroind)==0:
            raise ValueError('No library master Zero found for this image')                
        zerofile = cals['file'][zeroind[0]]
        if os.path.exists(zerofile):
            zezroim,zerohead = fits.getdata(zerofile,0,header=True)
        else:
            raise ValueError('Library '+zerofile+' file NOT FOUND')                   
    # Image input
    else:
        zeroim = zero
    # --- DARK ---    
    # Filename input
    if type(dark) is str:
        if os.path.exists(dark):
            darkim,darkhead = fits.getdata(dark,0,header=True)
        else:
            raise ValueError(dark+' NOT FOUND')
    # Use calibration library file
    elif dark is True:
        darkind, = np.where((cals['type']=='dark') & (cals['master']==True))
        if len(darkind)==0:
            raise ValueError('No library master Dark found for this image')                
        darkfile = cals['file'][darkind[0]]
        if os.path.exists(darkfile):
            darkim,darkhead = fits.getdata(darkfile,0,header=True)
        else:
            raise ValueError('Library '+darkfile+' file NOT FOUND')                   
    # Image input
    else:
        darkim = dark

    # FLATS are filter-specific

        
    # Image loop
    #-----------
    for i in range(nfiles):
        # Data input
        if len(files)==1 and files[0]=='':
            if head is None:
                raise ValueError('Header not input')
        # Load the data            
        else:
            if verbose:
                print('Loading '+files[i])
            if os.path.exists(files[i]):
                im,head = fits.getdata(files[i],0,header=True)
            else:
                raise ValueError(files[i]+' NOT FOUND')            
        
        # Fix header, if necessary
        if (head.get('TRIMSEC') is None) | (head.get('BIASSEC') is None):
            head = fixheader(head)
            
        # Overscan subtract and trim
        #---------------------------
        if head.get('OVERSCAN') is None:
            fim,fhead = overscan(im,head,verbose=verbose)
        else:
            print('Already OVERSCAN corrected')
            fim = im.copy()
            fhead = head.copy()
        fim = fim.astype(float)  # make sure it's float
            
        # Initialize error and mask image
        error = np.zeros(fim.shape,float)
        mask = np.zeros(fim.shape,np.uint8) # uint8, can handle values up to 255
        
        # Bad pixel mask
        #---------------
        if (bpm is not None):
            # Not corrected yet
            if head.get('BPMCOR') is None:            
                # Check sizes
                if bpmim.shape != fim.shape:
                    raise ValueError('BPM shape and image shape do not match')
                # Do the correction
                nbadbpm = np.sum(bpmim>0)
                if nbadbpm>0:
                    fim[bpmim>0] = 0.0
                    mask[bpmim>0] = 1
                    error[bpmim>0] = 1e30
                fhead['BPMCOR'] = time.ctime()+' BPM: masked '+str(nbadbpm)+' bad pixels'
                if verbose:
                    print(fhead['BPMCOR'])
            # Corrected already
            else:
                print('Already ZERO subtracted')
            
        # Set mask and error for saturated pixels
        #----------------------------------------
        saturation = head.get('saturate')
        if saturation is None:
            saturation = 64000
            sat = (fim>saturation) & (mask==0)
            mask[sat] = 2
            error[sat] = 1e30
    
        # Subtract master bias
        #---------------------
        if (zero is not None):
            # Not corrected yet
            if head.get('ZEROCOR') is None:
                # Check sizes
                if zeroim.shape != fim.shape:
                    raise ValueError('ZERO shape and image shape do not match')                
                # Do the correction
                fim[mask==0] -= zeroim[mask==0]
                fhead['ZEROCOR'] = time.ctime()+' ZERO: mean %6.2f, stdev %6.2f' % (np.mean(zeroim[mask==0]),np.std(zeroim[mask==0]))
                if verbose:
                    print(fhead['ZEROCOR'])
            # Corrected already
            else:
                print('Already ZERO subtracted')
            
        # Calculate error array
        #------------------------
        gain = head.get('gain')
        if gain is None:
            gain = 1.0
        rdnoise = head.get('rdnoise')
        if rdnoise is None:
            rdnoise = 0.0
        # Add Poisson noise and readnoise in quadrature
        error[mask==0] = np.sqrt(np.maximum(fim[mask==0]/gain,0)+rdnoise**2)
    
        # Subtract master dark scaled to this exposure time
        #--------------------------------------------------
        if (dark is not None):
            # Not corrected yet
            if head.get('DARKCOR') is None:   
                # Check sizes
                if darkim.shape != fim.shape:
                    raise ValueError('DARK shape and image shape do not match')                       
                # Do the correction
                fim[mask==0] -= darkim[mask==0]*head['exptime']
                fhead['DARKCOR'] = time.ctime()+' DARK: mean %6.2f, stdev %6.2f' % \
                                   (np.mean(darkim[mask==0]*head['exptime']),np.std(darkim[mask==0]*head['exptime']))
                if verbose:
                    print(fhead['DARKCOR'])
            # Corrected already
            else:
                print('Already DARK corrected')
            
        # Flat field
        #-----------
        if (flat is not None):
            # Not corrected yet
            if head.get('FLATCOR') is None:
                # Filename input
                if type(flat) is str:
                    if os.path.exists(flat):
                        flatim,flathead = fits.getdata(flat,0,header=True)
                    else:
                        raise ValueError(flat+' NOT FOUND')
                # Use calibration library file
                elif flat is True:
                    flatind, = np.where((cals['type']=='flat') & (cals['master']==True)
                                        & (cals['filter']==head['filter']))
                    if len(flatind)==0:
                        raise ValueError('No library master Flat found for this image')
                    flatfile = cals['file'][flatind[0]]
                    if os.path.exists(flatfile):
                        flatim,flathead = fits.getdata(flatfile,0,header=True)
                    else:
                        raise ValueError('Library '+flatfile+' file NOT FOUND')                   
                # Image input
                else:
                    flatim = flat
                # Check sizes
                if flatim.shape != fim.shape:
                    raise ValueError('FLAT shape and image shape do not match') 
                # Do the correction
                fim[mask==0] /= flatim[mask==0]
                error[mask==0] /= flatim[mask==0]  # need to divide error as well
                fhead['FLATCOR'] = time.ctime()+' FLAT: mean %6.2f, stdev %6.2f' % (np.mean(flatim[mask==0]),np.std(flatim[mask==0]))
                if verbose:
                    print(fhead['FLATCOR'])
            # Already corrected
            else:
                print('Already FLAT corrected')
                
        # Fix pix
        #--------
        # interpolate over bad pixels
        if fix:
            fim,mask,fhead = fixpix(fim,mask,fhead,verbose=verbose)
            
        fhead['CCDPROC'] = time.ctime()+' CCD processing done'
        if verbose:
            print(time.ctime()+' CCD processing done')

        # Convert images
        fim = fim.astype(np.float32)     # to 32 bit
        error = error.astype(np.float32) # to 32 bit
        
        # Write to output file
        if outfile is not None:
            # Use automatic name with suffix
            if outfile is True:
                if files[i] != '':
                    outdir = os.path.dirname(files[i])
                    if outdir=='': outdir='.'
                    outbase = os.path.basename(files[i])
                    if '.gz' in outbase:
                        outbase = outbase[:-3]
                    outbase,outext = os.path.splitext(outbase)
                    outfil = outdir+'/'+outbase+outsuffix+'.fits'
                else:
                    outfil = 'inputimage'+outsuffix+'.fits'
            else:
                outfil = outfile
            # Check if the output file exists already
            if os.path.exists(outfil):
                if clobber is False:
                    raise ValueError(outfil+' already exists and clobber=False')
                else:
                    os.remove(outfil)
            print('Writing processed file to '+outfil)
            hdulist = fits.HDUList()
            hdu = fits.PrimaryHDU(fim,fhead)
            hdulist.append(hdu)
            # Add error image
            hdulist.append(fits.ImageHDU(error))
            hdulist[1].header['BUNIT'] = 'error'
            # Add mask image
            hdulist.append(fits.ImageHDU(mask))
            hdulist[2].header['BUNIT'] = 'mask'
            hdulist[2].header['HISTORY'] = ' Mask values'
            hdulist[2].header['HISTORY'] = ' 0: good'        
            hdulist[2].header['HISTORY'] = ' 1: bad pixel'
            hdulist[2].header['HISTORY'] = ' 2: saturated'
            hdulist[2].header['HISTORY'] = ' 4: interpolated'
            hdulist.writeto(outfil,overwrite=clobber)
            hdulist.close()
            # Gzip compress
            if compress:
                if verbose:
                    print('Gzip compressing')
                if os.path.exists(outfil+'.gz') and clobber:
                    os.remove(outfil+'.gz')
                    out = subprocess.run(['gzip',outfil])
                else:
                    print(outfil+'.gz already exists and clobber=False')
        
    return fim, fhead

def redrun(files):
    """
    Automatically reduce an entire run of data.
    """

    tab = ccdlist(files)
    # Step 1) Reduce bias/zeros
    #--------------------------
    
    # Step 2) Make master zero
    #--------------------------
    
    # Step 3) Reduce darks
    #---------------------
    
    # Step 4) Make master dark
    #-------------------------
    
    # Step 5) Reduce flats
    #---------------------
    
    # Step 6) Make master flat
    #-------------------------
    
    # Step 7) Reduce science/object exposures
    #----------------------------------------
    
    pass

def autored(datadir='.'):
    """ Automatically pick up FITS files from a directory and reduce."""

    # While loop
    count = 0
    wait = 10
    flag = 0
    lastfiles = []
    while (flag==0):
        # Check directory for new files
        files = glob(datadir+'/*.fit*')
        nfiles = len(files)
        
        # If there are new files then reduce them
        if files!=lastfiles:
            newfiles = [f for f in files if f not in lastfiles]
            nnewfiles = len(newfiles)
            print(time.ctime()+' Found '+str(nnewfiles)+' new files: '+','.join(newfiles))
            for i in range(nnewfiles):
                base = os.path.basename(newfiles[i])
                outfile = datadir+'/red/'+base
                out = ccdproc(newfiles[i],outfile=outfile,compress=True)
        
        # Sleep for a while
        time.sleep(wait)

        # Last list of files
        lastfiles = files
