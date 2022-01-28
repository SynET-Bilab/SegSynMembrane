""" refine
usage:
    dO, mask = diff_fit_seg(mootools, indiv, O_seg, sigma_gauss, sigma_tv)
    B_filt = filter_seg(mootools.B, dO, mask, factor)
"""
import numpy as np
import skimage
from synseg import hessian
from synseg import dtvoting
from synseg import utils

__all__ = [
    "find_max_wire", "diff_fit_seg",
    "gaussian_pdf", "mixture_gaussian",
    "filter_seg"
]

#=========================
# orientation differences
#=========================

def find_max_wire(pts_net, axis):
    # A, B - axes
    # [dz,dy,dx] along A for each B
    diff_zyx = np.diff(pts_net, axis=axis)
    # len of wire segments along A for each B
    segments = np.linalg.norm(diff_zyx, axis=-1)
    # len of wire along A for each B
    wire = np.sum(segments, axis=axis)
    # max of wire in all B
    wire_max = np.max(wire)
    return wire_max

def diff_fit_seg(mootools, indiv, O_seg, sigma_gauss, sigma_tv):
    """ calculate difference of orientation between fit and segmentation from tv->filter->connected
    :param mootools: MOOTools
    :param indiv: individual from MOOPop
    :param O_seg: orientation from prev segmentation
    :param sigma_gauss: sigma for calculate hessian
    :param sigma_tv: sigma for tensor voting on fitted surface
    :return: dO, mask
        dO: difference of orientation, shape=(nz,ny,nx), range=(0,pi/2)
        mask: bool, shape=(nz,ny,nx), pixels belong to B_seg and close to fit
    """
    # generate binary image from fit
    # generate eval pts: max of wireframes
    pts_net = mootools.get_coord_net(indiv)
    n_eval_uz = find_max_wire(pts_net, axis=0)
    n_eval_vxy = find_max_wire(pts_net, axis=1)

    # fit at dense eval pts
    Bfit, _ = mootools.fit_surface_eval(
        indiv,
        u_eval=np.linspace(0, 1, 2*int(n_eval_uz)),
        v_eval=np.linspace(0, 1, 2*int(n_eval_vxy))
    )

    # TV for fit
    _, Ofit = hessian.features3d(Bfit, sigma=sigma_gauss)
    Sfit_tv, Ofit_tv = dtvoting.stick3d(Bfit, Ofit*Bfit, sigma=sigma_tv)

    # mask: belongs to B and close to Bfit
    # e^(-1/2) = e^(-r^2/(2*sigma_tv^2)) at r=sigma_tv, close to fill holes
    mask_fit = Sfit_tv > np.exp(-1/2)
    mask_fit = skimage.morphology.binary_closing(mask_fit)

    # difference in orientation
    dO = utils.absdiff_orient(O_seg, Ofit_tv)
    dO = dO*mask_fit*mootools.B
    return dO, mask_fit

#=========================
# Gaussian mixture model
#=========================
class GMMFixed:
    """ Gaussian mixture model with fixed means
    """
    def __init__(self, means_fixed=(0, np.pi/2),
        sigmas_init=(0.1, 0.1), weights_init=(0.5, 0.5),
        tol=0.001, max_iter=100,
        ):
        """ init, order: (signal, noise)
        :param means_fixed: means to be fixed
        :param sigmas_init, weights_init: init params
        :param tol, max_iter: iteration controls
        """
        self.means = means_fixed
        self.sigmas = sigmas_init
        self.weights = weights_init
        self.tol = tol
        self.max_iter = max_iter

    def gauss(self, x, sigma):
        return np.exp(-x**2/(2*sigma**2))/((2*np.pi)**0.5*sigma)

    def posterior(self, x):
        """ calculate posterior
        :param x: 1d data array
        :return: p_post
            p_post: shape=(2, len(x)), p_post[i] is posterior for category i
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
        """ maximize parameters
        :param x: 1d data array
        :param p_post: posterior
        :return: weights, sigmas
        """
        weights = tuple(np.mean(p_post[i]) for i in range(2))
        sigmas = tuple(
            np.sqrt(np.sum(p_post[i]*(x-self.means[i])**2)/np.sum(p_post[i]))
            for i in range(2)
        )
        return weights, sigmas

    def fit_threshold(self, x):
        """ fit data, calculate threshold
        :param x: 1d data array
        :return: x_thresh
            x_thresh: signals are where x < x_thresh
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
# filter by orientation
#=========================

def filter_seg(B_seg, dO, mask):
    """ filter segmented image by dO
    :param B_seg: binary image from prev segmentation
    :param dO, mask: results from diff_fit_seg()
    :param factor: dO_threshold = factor*dO_sigma, dO_sigma is from mixture_gaussian()
    :return: B_filt
        B_filt: filtered B_seg
    """
    # estimate peak
    gmm = GMMFixed(means_fixed=(0, np.pi/2))
    dO_thresh = gmm.fit_threshold(dO[mask])

    # filter
    B_filt = B_seg * (dO < dO_thresh) * mask
    return B_filt
