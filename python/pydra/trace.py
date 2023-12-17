#!/usr/env python

# Imports
import numpy as np
from scipy.special import erf,wofz
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks,argrelextrema,convolve2d
from scipy.ndimage import median_filter,uniform_filter
from astropy.io import fits
from scipy import ndimage
from scipy.interpolate import interp1d
from numba import njit
from dlnpyutils import utils as dln,robust,mmm
import matplotlib.pyplot as plt
#from . import utils,robust,mmm

class Traces():
    """
    Class to hold multiple spectral traces.
    """

    def __init__(self,data=None):
        # Input a trace list or an image
        self._data = None
        if data is not None:
            if type(data) is list:
                self._data = data
            elif type(data) is numpy.ndarray and data.ndim==2:
                tlist = findtrace(im)
                self._data = tlist
            # Sort them
            
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None        

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
            return Trace(self._data[item])
        # Multiple traces
        elif type(item) is tuple or type(item) is slice:
            index = np.arange(len(self))
            index = index[item]  # apply it to the indices
            data = len(index)*[None]
            for i in range(len(index)):
                data[i] = self._data[index[i]]  # by reference
            return Traces(data)
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

    def __call__(self,x=None):
        """ Return the trace path."""
        if self._data is None:
            raise ValueError('No trace data yet')
        if x is None:
            x = np.arange(self._data['xmin'],self._data['xmax'])
        # Get the curve/path
        y = np.polyval(self._data['tcoef'],x)
        out = np.zeros(len(x),dtype=np.dtype([('x',float),('y',float)]))
        out['x'] = x
        out['y'] = y
        return out     

    def model(self,x=None,yr=None):
        """ Return normalized 2D image model."""
        if x is None:
            xr = [int(np.ceil(self.xmin)),int(np.floor(self.xmax))]
            x = np.arange(xr[1]-xr[0]+1)+xr[0]
        if yr is None:
            sigma = np.median(self.sigma)
            yr = [int(np.floor(np.min(self.y)-3*sigma)),
                  int(np.ceil(np.max(self.y)+3*sigma))]
        y = np.arange(yr[1]-yr[0]+1)+yr[0]
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
        return self()['y']

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
        return np.min(self()['y'])

    @property
    def ymax(self):
        if self.hasdata==False:
            return None
        return np.max(self()['y'])        
    
    @property
    def sigma(self):
        if self.hasdata==False:
            return None
        return self._data['gysigma']
    
    def fit(self,image,initpos=None):
        """ Fit a trace to data."""
        if initpos is not None:
            if len(initpos)!=2:
                raise ValueError("Initial guess position must have 2 elements (X,Y)")
        # Find tracing in the image
        trlist = findtrace(image)
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

    def copy(self):
        return Trace(self._data.copy())
        
    def plot(self,**kwargs):
        out = self()
        plt.plot(out['x'],out['y'],**kwargs)

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
def gpars(x,y):
    """
    Simple Gaussian fit to central 5 pixel values.

    Parameters
    ----------
    x : numpy array
       Numpy array of x-values.
    y : numpy array
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
    
    if y.ndim==1:
        nprofiles = 1
        y = np.atleast_1d(y)
    else:
        nprofiles = y.shape[0]
    nx = y.shape[1]
    nhalf = nx//2
    # Loop over profiles
    pars = np.zeros((nprofiles,3),float)
    for i in range(nprofiles):
        x1 = x[i,:]
        y1 = y[i,:]
        gd = (np.isfinite(y1) & (y1>0))
        if np.sum(gd)<5:
            x1 = x1[gd]
            y1 = y1[gd]
        totflux = np.sum(y1)
        ht0 = y[i,nhalf]
        if np.isfinite(ht0)==False:
            ht0 = np.max(y1)
        # Use flux-weighted moment to get center
        cen = np.sum(y1*x1)/totflux
        #  Gaussian area is A = ht*wid*sqrt(2*pi)
        sigma = np.maximum( totflux/(ht0*np.sqrt(2*np.pi)) , 0.01)
        # Use linear-least squares to calculate height and sigma
        psf = np.exp(-0.5*(x1-cen)**2/sigma**2)          # normalized Gaussian
        wtht = np.sum(y1*psf)/np.sum(psf*psf)          # linear least squares
        pars[i,:] = [wtht,cen,sigma]
        
    return pars

def backvals(backpix):
    """ Estimate the median and sigma of background pixels."""
    vals = backpix.copy()
    medback = np.nanmedian(vals)
    sigback = dln.mad(vals)
    lastmedback = 1e30
    lastsigback = 1e30
    done = False
    # Outlier rejection
    while (done==False):
        gdback, = np.where(vals < (medback+3*sigback))
        vals = vals[gdback]
        medback = np.nanmedian(vals)
        sigback = dln.mad(vals)
        if np.abs(medback-lastmedback) < 0.01*lastmedback and \
           np.abs(sigback-lastsigback) < 0.01*lastsigback:
            done = True
        lastmedback = medback
        lastsigback = sigback
    return medback,sigback

def findtrace(im,nbin=50,minsigheight=3,minheight=None,hratio2=0.97,
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

    tracelist = findtrace(im)

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
    profiles = np.zeros((nprofiles,5),float)
    xprofiles = np.zeros((nprofiles,5),int)
    count = 0
    for tr in tracelist:
        for i in range(tr['ncol']):
            profiles[count,:] = medim[tr['yvalues'][i]-2:tr['yvalues'][i]+3,tr['xbvalues'][i]]
            xprofiles[count,:] = np.arange(5)+tr['yvalues'][i]-2
            count += 1
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

    # Add YMED and sort
    for i in range(len(tracelist)):
        itrace = tracelist[i]
        itrace['ymed'] = np.median(itrace['yvalues'])
        tracelist[i] = itrace
    # Sort them
    tracelist.sort(key=lambda t:t['ymed'])
        
    return tracelist
    

def trace(im,yestimate=None,yorder=2,sigorder=2,step=50):
    """ Trace the spectrum.  Spectral dimension is assumed to be on the horizontal axis."""
    ny,nx = im.shape
    y = np.arange(ny)
    if yestimate is None:
        ymed = np.nanmedian(im,axis=1)
        #yestimate = np.argmax(ytot)
        sig = dln.mad(ymed)
        ypeaks, = argrelextrema(ymed, np.greater)
        gd, = np.where(ymed[ypeaks] > (np.nanmedian(ymed)+5*sig))
        ypeaks = ypeaks[gd]
        #ypeaks,ptab = find_peaks(ymed,height=10*sig,width=1,prominence=10*sig,wlen=100,distance=10)
        print(len(ypeaks),' peaks found')
    else:
        ypeaks = yestimate
        print(len(ypeaks),' peaks input')
        
    # Smooth in spectral dimension
    # a uniform (boxcar) filter with a width of 50
    smim = ndimage.uniform_filter1d(im, 50, 1)
    nstep = nx//step
    out = np.zeros(len(ypeaks),dtype=np.dtype([('peak',int),('yestimate',int),('tpars',(float,yorder+1)),
                                               ('sigpars',(float,sigorder+1))]))
    mcat = []
    # Loop over the traces
    for t in range(len(ypeaks)):
        yestimate = ypeaks[t]
        # Loop over the columns in steps and fit Gaussians
        tcat = np.zeros(nstep,dtype=np.dtype([('x',float),('pars',float,4),('status',int)]))
        for i in range(nstep):
            try:
                ht = np.maximum(im[yestimate,step*i+step//2],100)
                estimate = [ht,yestimate,2.0,0.0]
                bounds = [np.zeros(4,float)-np.inf,np.zeros(4,float)+np.inf]
                bounds[0][0] = 0
                bounds[0][1] = yestimate-15
                bounds[1][1] = yestimate+15                
                bounds[0][2] = 0.1
                pars,cov = dln.gaussfit(y[yestimate-15:yestimate+15],im[yestimate-15:yestimate+15,step*i+step//2],
                                          initpar=estimate,bounds=bounds,binned=True)
                tcat['x'][i] = step*i+step//2
                tcat['pars'][i] = pars
                tcat['status'][i] = 1
            except:
                pass
        # Fit polynomial to y vs. x and gaussian sigma vs. x
        gd, = np.where(tcat['status']==1)
        ypars = np.polyfit(tcat['x'][gd],tcat['pars'][gd,1],yorder)
        sigpars = np.polyfit(tcat['x'][gd],tcat['pars'][gd,2],sigorder)
        print(t+1,yestimate,pars,sigpars)
        # Model
        mcat1 = np.zeros(nx,dtype=np.dtype([('x',float),('y',float),('sigma',float)]))
        xx = np.arange(nx)
        mcat1['x'] = xx
        mcat1['y'] = np.poly1d(ypars)(xx)
        mcat1['sigma'] = np.poly1d(sigpars)(xx)
        mcat.append(mcat1)
        # output
        out['peak'][t] = t
        out['yestimate'][t] = yestimate
        out['tpars'][t] = ypars
        out['sigpars'][t] = sigpars      
        
    return out,mcat

def boxcar(im,ytrace=None,imerr=None,off=20,backoff=50):
    """ Boxcar extract the spectrum"""
    ny,nx = im.shape
    if ytrace is None:
        ytot = np.sum(im,axis=1)
        yest = np.argmax(ytot)
        ytrace = np.arange(nx)+yest
    else:
        yest = np.median(ytrace)
    # Background subtract
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    med = np.median(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    # Sum up the flux
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    flux = np.sum(subim*mask,axis=0)
    if imerr is not None:
        # add uncertainties in quadrature
        fluxerr = np.sqrt(np.sum(imerr[yblo:ybhi,:]**2*mask,axis=0))
        return flux,fluxerr
    return flux

def linefit(x,y,initpar,bounds,err=None,binned=False):
    # Fit Gaussian profile to data with center and sigma fixed.
    # initpar = [height, center, sigma, constant offset]
    cen = initpar[1]
    sigma = initpar[2]
    #def gline(x, amp, const=0):
    #    """1-D gaussian: gaussian(x, amp, cen, sig)"""
    #    return amp * np.exp(-(x-cen)**2 / (2*sigma**2)) + const
    #line_initpar = [initpar[0],initpar[3]]
    #lbounds, ubounds = bounds
    #line_bounds = ([lbounds[0],lbounds[3]],[ubounds[0],ubounds[3]])
    #return curve_fit(gline, x, y, p0=line_initpar, bounds=line_bounds, sigma=err)

    func = dln.gaussian
    if binned is True: func=dln.gaussbin
    line_initpar = [initpar[0],initpar[3]]
    line_bounds = ([lbounds[0],lbounds[3]],[ubounds[0],ubounds[3]])
    return curve_fit(func, x, y, p0=line_initpar, sigma=sigma, bounds=line_bounds)
    
    return curve_fit(gline, x, y, p0=line_initpar, bounds=line_bounds, sigma=err)

def extract(im,imerr=None,mcat=None,fixtrace=False,fixsigma=False,nobackground=False,
            verbose=False,off=10):
    """ Extract a spectrum"""
    ny,nx = im.shape
    x = np.arange(nx)
    y = np.arange(ny)
    # No trace information input, get it
    if mcat is None:
        tcat,ypars,sigpars,mcat=trace(im)
    # Loop over the columns and get the flux using the trace information
    tab = np.zeros(nx,dtype=np.dtype([('x',int),('pars',float,4),('perr',float,4),
                                      ('flux',float),('fluxerr',float),('chisq',float),('status',int)]))
    #lastpars = None
    for i in range(nx):
        line = im[:,i].copy()
        if imerr is not None:
            lineerr = imerr[:,i].copy()
        else:
            lineerr = np.ones(len(line))   # unweighted        
        bad = (~np.isfinite(line)) | (lineerr==0)
        if np.sum(bad)>0:
            med = np.nanmedian(line[~bad])
            if ~np.isfinite(med): med=0
            line[bad] = med
            lineerr[bad] = 1e30
        # Fit the constant offset and the height of the Gaussian
        #  fix the central position and sigma
        ycen = mcat['y'][i]
        ysigma = mcat['sigma'][i]
        ht0 = np.maximum(line[int(np.round(ycen))],0.01)
        initpars = [ht0,ycen,ysigma,np.median(line)]
        #if lastpars is not None:
        #    initpars = lastpars
        if nobackground is True:
            initpars[3] = 0
        # Only fit the region right around the peak
        ylo = int(np.maximum(ycen-off,0))
        yhi = int(np.minimum(ycen+off,ny))
        y1 = y[ylo:yhi]
        line1 = line[ylo:yhi]
        err1 = lineerr[ylo:yhi]
        #bnds = ([0,ycen-1e-4,ysigma-1e-7,np.minimum(0,initpar[3]-1)],
        #        [1.5*ht0,ycen,ysigma+1e-7,np.maximum(1.5*initpar[3],initpar[3]+1)])
        bnds = (np.zeros(4)-np.inf,np.zeros(4)+np.inf)
        bnds[0][0] = 0   # height
        bnds[0][1] = initpars[1]-2
        bnds[1][1] = initpars[1]+2        
        if fixtrace:
            bnds[0][1] = initpars[1]-1e-7  # fix the trace position
            bnds[1][1] = initpars[1]+1e-7
        bnds[0][2] = 0     # positive sigma
        if fixsigma:
            bnds[0][2] = initpars[2]-1e-7
            bnds[1][2] = initpars[2]+1e-7
        # background
        bnds[0][3] = np.minimum(0,initpars[3]-1)
        bnds[1][3] =  np.maximum(1.5*initpars[3],initpars[3]+1)
        if nobackground is True:
            bnds[0][3] = initpars[3]-1e-7
            bnds[1][3] = initpars[3]+1e-7
            #bnds = ([0,ycen-1e-7,ysigma-1e-7,0],[1.5*ht0,ycen+1e-7,ysigma+1e-7,0.1])

        #func = dln.gaussian
        #if binned is True: func=dln.gaussbin
        func = dln.gaussbin
        try:
            pars,cov = curve_fit(func,y1,line1,p0=initpars,bounds=bnds,sigma=err1)
            # reject outlier points and refit
            perr = np.sqrt(np.diag(cov))
            yfit = func(y1,*pars)
            diff = y1-yfit
            bd, = np.where(np.abs(diff) > 3*err1)
            if len(bad)>0:
                pars1 = pars
                line1[bd] = yfit[bd]
                err1[bd] = 1e30
                pars,cov = curve_fit(func,y1,line1,p0=pars1,bounds=bnds,sigma=err1)                   
                # reject outlier points and refit
                perr = np.sqrt(np.diag(cov))
                yfit = func(y1,*pars)                   
            #pars,cov = linefit(y[y0:y1],line[y0:y1],initpar=initpar,bounds=bnds,err=lineerr[y0:y1])
            flux = np.sum(yfit)
            # Gaussian area = ht*wid*sqrt(2*pi)
            #flux = pars[0]*ysigma*np.sqrt(2*np.pi)
            #fluxerr = perr[0]*ysigma*np.sqrt(2*np.pi)
            # propagation of errors
            fluxerr = flux*np.sqrt( (perr[0]/pars[0])**2 + (perr[2]/pars[2])**2 )
            chisq = np.sum((line1-yfit)/err1)
            if verbose:
                print(i+1,pars)
            tab['x'][i] = i
            tab['pars'][i] = pars
            tab['perr'][i] = perr
            tab['flux'][i] = flux
            tab['fluxerr'][i] = fluxerr
            tab['chisq'][i] = chisq
            tab['status'][i] = 1
            #lastpars = pars
        except:
            if verbose:
                print(i+1,' problem fitting')
    return tab

def extract_optimal(im,ytrace,imerr=None,verbose=False,off=10,backoff=50,smlen=31):
    """ Extract a spectrum using optimal extraction (Horne 1986)"""
    ny,nx = im.shape
    yest = np.median(ytrace)
    # Get the subo,age
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    # Background subtract
    med = np.median(im[yblo:ybhi,:],axis=0)
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
    tot = np.sum(np.maximum(sim,0),axis=0)
    tot[(tot<=0) | ~np.isfinite(tot)] = 1
    psf1 = np.maximum(sim,0)/tot
    psf = np.zeros(psf1.shape,float)
    for i in range(nback):
        psf[i,:] = dln.medfilt(psf1[i,:],smlen)
        #psf[i,:] = dln.gsmooth(psf1[i,:],smlen)        
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    totpsf = np.sum(psf,axis=0)
    totpsf[(totpsf<=0) | (~np.isfinite(totpsf))] = 1
    psf /= totpsf
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    # Compute the weights
    wt = psf**2/serr**2
    wt[(wt<0) | ~np.isfinite(wt)] = 0
    totwt = np.sum(wt,axis=0)
    badcol = (totwt<=0)
    totwt[badcol] = 1
    # Compute the flux and flux error
    flux = np.sum(psf*sim/serr**2,axis=0)/totwt
    fluxerr = np.sqrt(1/totwt)    
    fluxerr[badcol] = 1e30  # bad columns
    # Recompute the trace
    trace = np.sum(psf*yy,axis=0)+yblo
    
    # Check for outliers
    diff = (sim-flux*psf)/serr**2
    bad = (diff > 25)
    if np.sum(bad)>0:
        # Mask bad pixels
        sim[bad] = 0
        serr[bad] = 1e20
        # Recompute the flux
        wt = psf**2/serr**2
        totwt = np.sum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1        
        flux = np.sum(psf*sim/serr**2,axis=0)/totwt
        fluxerr = np.sqrt(1/totwt)
        fluxerr[badcol] = 1e30  # bad columns
        # Recompute the trace
        trace = np.sum(psf*yy,axis=0)+yblo
        
    return flux,fluxerr,trace
        
def emissionlines(spec,thresh=None):
    """Measure the emission lines in an arc lamp spectrum. """
    nx = len(spec)
    x = np.arange(nx)
    
    # Threshold
    if thresh is None:
        thresh = np.min(spec) + (np.max(spec)-np.min(spec))*0.05
    
    # Detect the peaks
    sleft = np.hstack((0,spec[0:-1]))
    sright = np.hstack((spec[1:],0))
    peaks, = np.where((spec>sleft) & (spec>sright) & (spec>thresh))
    npeaks = len(peaks)
    print(str(npeaks)+' peaks found')
    
    # Loop over the peaks and fit them with Gaussians
    gcat = np.zeros(npeaks,dtype=np.dtype([('x0',int),('x',float),('xerr',float),('pars',float,4),('perr',float,4),
                                           ('flux',float),('fluxerr',float)]))
    resid = spec.copy()
    gmodel = np.zeros(nx)
    for i in range(npeaks):
        x0 = peaks[i]
        xlo = np.maximum(x0-6,0)
        xhi = np.minimum(x0+6,nx)
        initpar = [spec[x0],x0,1,0]
        bnds = ([0,x0-3,0.1,0],[1.5*initpar[0],x0+3,10,1e4])
        pars,cov = dln.gaussfit(x[xlo:xhi],spec[xlo:xhi],initpar,bounds=bnds,binned=True)
        perr = np.sqrt(np.diag(cov))
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
        
    return gcat, gmodel


def continuum(spec,bin=50,perc=60,norder=4):
    """ Derive the continuum of a spectrum."""
    nx = len(spec)
    x = np.arange(nx)
    # Loop over bins and find the maximum
    nbins = nx//bin
    xbin1 = np.zeros(nbins,float)
    ybin1 = np.zeros(nbins,float)
    for i in range(nbins):
        xbin1[i] = np.nanmean(x[i*bin:i*bin+bin])
        ybin1[i] = np.nanpercentile(spec[i*bin:i*bin+bin],perc)
    # Fit polynomial to the binned values
    coef1 = robust.polyfit(xbin1,ybin1,norder)
    cont1 = np.poly1d(coef1)(x)
    
    # Now remove large negative outliers and refit
    gdmask = np.zeros(nx,bool)
    gdmask[(spec/cont1)>0.8] = True
    xbin = np.zeros(nbins,float)
    ybin = np.zeros(nbins,float)
    for i in range(nbins):
        xbin[i] = np.nanmean(x[i*bin:i*bin+bin][gdmask[i*bin:i*bin+bin]])
        ybin[i] = np.nanpercentile(spec[i*bin:i*bin+bin][gdmask[i*bin:i*bin+bin]],perc)
    # Fit polynomial to the binned values
    coef = robust.polyfit(xbin,ybin,norder)
    cont = np.poly1d(coef)(x)
    
    return cont,coef
