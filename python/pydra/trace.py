#!/usr/env python

# Imports
import numpy as np
import time
from scipy.special import erf,wofz
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks,argrelextrema,convolve2d
from scipy.ndimage import median_filter,uniform_filter
from astropy.io import fits
from scipy import ndimage
from scipy.interpolate import interp1d
from numba import njit
from dlnpyutils import utils as dln,robust,mmm,coords
from matplotlib.path import Path
import matplotlib.pyplot as plt
#from . import utils,robust,mmm
from . import extract as xtract

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
    
    @property
    def sigma(self):
        if self.hasdata==False:
            return None
        return self._data['gysigma']

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
def gpars1(x,y):
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
    lny = np.log(y)
    quadcoef = quadratic_coefficients(x-cen1,lny)
    # a = -1/(2*sigma**2)   ->   sigma=sqrt(-1/(2*a))
    # b = x0/sigma**2       ->   x0=b*sigma**2
    # c = -x0**2/(2*sigma**2) + lnA  ->  A=exp(c + x0**2/(2*sigma**2))
    sigma = np.sqrt(-1/(2*quadcoef[0]))
    x0 = quadcoef[1]*sigma**2
    height = np.exp(quadcoef[2]+x0**2/(2*sigma**2))
    
    pars[:] = [height,cen1+x0,sigma] 
    return pars
    
@njit
def gpars(x,y):
    """
    Simple Gaussian fit to central pixel values.

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
        # First try the central 5 pixels first        
        x1 = x[i,2:7]
        y1 = y[i,2:7]
        gd = (np.isfinite(y1) & (y1>0))
        if np.sum(gd)<5:
            x1 = x1[gd]
            y1 = y1[gd]
        pars1 = gpars1(x1,y1)
        # If sigma is too high, then expand to include more points     
        if pars1[2]>2:
            x1 = x[i,:]
            y1 = y[i,:]
            gd = (np.isfinite(y1) & (y1>0))
            if np.sum(gd)<9:
                x1 = x1[gd]
                y1 = y1[gd]
            pars1 = gpars1(x1,y1)            
        # Put results in big array
        pars[i,:] = pars1
        
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
    count = 0
    for tr in tracelist:
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

    # Should we check for overlap of traces?
        
    # Add YMED and sort
    for i in range(len(tracelist)):
        itrace = tracelist[i]
        itrace['ymed'] = np.median(itrace['yvalues'])
        tracelist[i] = itrace
    # Sort them
    tracelist.sort(key=lambda t:t['ymed'])
        
    return tracelist
