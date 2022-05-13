""" Matching: select points from the detected which match the fitted surface.
"""

import numpy as np
import skimage
from etsynseg import imgutils
from etsynseg import features, dtvoting, tracing

__all__ = [
    "GMMFixed", "fill_broken", "match_spatial_orient"
]

#=========================
# GMM
#=========================

class GMMFixed:
    """ Gaussian mixture model (GMM) with fixed means.
    """
    def __init__(self, means_fixed=(0, np.pi/2),
        sigmas_init=(0.1, 0.1), weights_init=(0.5, 0.5),
        tol=0.001, max_iter=100,
        ):
        """ Initialize the GMM.
        
        Args:
            means_fixed (2-tuple): Fix means to this value, (mean_signal,mean_noise).
            sigmas_init (2-tuple): Initial (sigma_signal,sigma_noise).
            weights_init (2-tuple): Initial (weight_signal,weight_noise).
            tol (float): Tolerance for convergence.
            max_iter (int): Max number of iterations.
        """
        self.means = means_fixed
        self.sigmas = sigmas_init
        self.weights = weights_init
        self.tol = tol
        self.max_iter = max_iter

    def gauss(self, x, sigma):
        """ Gaussian function, e^(-x^2/2sigma^2)/(sqrt(2pi)*sigma)

        Args:
            x, sigma (float): Parameters of the function.
        """
        return np.exp(-x**2/(2*sigma**2))/((2*np.pi)**0.5*sigma)

    def posterior(self, x):
        """ Posterior calculation, E-step.

        Args:
            x (np.ndarray): 1d array for data.

        Returns:
            p_post (np.ndarray): Shape=(2,len(x)). p_post[i] is the posterior for category i (0 for signal, 1 for noise).
        """
        # conditional probabilities
        p_cond = np.empty((2, len(x)))
        for i in range(2):
            p_cond[i] = self.weights[i]*self.gauss((x-self.means[i]), self.sigmas[i])

        # posterior probabilities
        p_post_sum = np.sum(p_cond, axis=0)
        p_post = np.empty((2, len(x)))
        for i in range(2):
            p_post[i] = p_cond[i] / p_post_sum
        return p_post

    def maximize_params(self, x, p_post):
        """ Maximize parameters, M-step.

        Args:
            x (np.ndarray): 1d array for data.
            p_post (np.ndarray): Posterior with shape=(2,len(x)).

        Returns:
            weights (np.ndarray): Shape=(2,len(x)). Weight parameters for (signal,noise).
            sigmas (np.ndarray): Shape=(2,len(x)). Sigma parameters for (signal,noise).
        """
        weights = tuple(np.mean(p_post[i]) for i in range(2))
        sigmas = tuple(
            np.sqrt(np.sum(p_post[i]*(x-self.means[i])**2)/np.sum(p_post[i]))
            for i in range(2)
        )
        return weights, sigmas

    def fit_threshold(self, x):
        """ Fit data. Calculate a threshold value that separates signal and noise.

        Args:
            x (np.ndarray): 1d array for data.

        Returns:
            x_thresh (float): Considered x<x_thresh as signals.
        """
        for i in range(self.max_iter):
            # E-step
            if i == 0:
                p_post = self.posterior(x)
            else:
                p_post = p_post_new

            # M-step
            self.weights, self.sigmas = self.maximize_params(x, p_post)

            # termination
            p_post_new = self.posterior(x)
            if np.max(p_post_new-p_post) < self.tol:
                break

        # get threshold
        p_post = self.posterior(x)
        x_thresh = np.max(x[p_post[0] > 0.5])
        return x_thresh


#=========================
# match by orientation
#=========================

def fill_broken(Bseg, Oseg, Bmatch):
    """ Fill broken segments of the matched image.

    For a connected line segment of a 2d slice,
    if two of its regions are matched but their middle part is missed,
    this function will fill the missing parts.

    Args:
        Bseg, Oseg (np.ndarray): Detected binary image and orientation, with shape=(nz,ny,nx).
        Bmatch (np.ndarray): Matched parts of the detected image, with shape=(nz,ny,nx).

    Returns:
        Bfill (np.ndarray): Filled image, with shape=(nz,ny,nx).
    """
    # setup
    tr = tracing.Tracing(Bseg, Oseg)
    Bfill = np.zeros(Bseg.shape, dtype=np.int_)

    # fix one slice
    def calc_one(iz):
        # get traces using depth-first scan
        yx_trs, _ = tr.dfs2d(iz)
        for yx_tr_i in yx_trs:
            # positions of pixels in the trace
            pos = tuple(np.transpose(yx_tr_i))
            # corresponding pixels in matched image
            b_match = Bmatch[iz][pos]
            i_match = np.nonzero(b_match)[0]
            # fill zeros inside the matched image
            if len(i_match) > 0:
                tr_filled = np.zeros(len(pos[0]), dtype=np.int_)
                tr_filled[np.min(i_match):np.max(i_match)+1] = 1
                Bfill[iz][pos] = tr_filled

    # fix all slices
    for iz in range(Bseg.shape[0]):
        calc_one(iz)

    return Bfill

def match_spatial_orient(Bseg, Oseg, Bfit, sigma_hessian, sigma_tv):
    """ Match the detected image with the fitted one by spatial and orientational closeness.
    
    First apply hessian+TV to the fitted surface to extend it to nearby regions.
    Compare the orientations between the extended fitted surface and the detected image, and threshold by GMM.
    Raw matched pixels in the detected image are spatially and orientationally close to the fitted one.
    Fill broken segments and get the final matched image.
    
    Args:
        Bseg, Oseg (np.ndarray): Detected binary image and orientation, with shape=(nz,ny,nx).
        Bfit (np.ndarray): Binary image of the surface from EvoMSAC fitting.
        sigma_hessian (float): Sigma for ridgelike feature calculation of the fitted surface.
            Can set to the same value as the one used for detection.
        sigma_tv (float): Sigma for TV of the fitted surface.
    
    Returns:
        Bmatch (np.ndarray): Matched binary image, with shape=(nz,ny,nx).
    """
    # spatial closeness: TV for fit
    _, Ofit = features.ridgelike3d(Bfit, sigma=sigma_hessian)
    Sfit_tv, Ofit_tv = dtvoting.stick3d(Bfit, Ofit*Bfit, sigma=sigma_tv)

    # mask: belongs to B and close to Bfit
    # e^(-1/2) = e^(-r^2/(2*sigma_tv^2)) at r=sigma_tv, close to fill holes
    mask_fit = Sfit_tv > np.exp(-1/2)
    mask_fit = skimage.morphology.binary_closing(mask_fit)
    mask = (mask_fit*Bseg).astype(bool)

    # difference in orientation
    dO = imgutils.orients_absdiff(Oseg, Ofit_tv)

    # estimate dO threshold using GMM
    gmm = GMMFixed(means_fixed=(0, np.pi/2))
    dO_thresh = gmm.fit_threshold(dO[mask])

    # match by threshold
    Bmatch = Bseg * (dO < dO_thresh) * mask

    # fill broken parts
    Bmatch = fill_broken(Bseg, Oseg, Bmatch)

    return Bmatch

