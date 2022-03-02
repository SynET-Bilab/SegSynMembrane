import numpy as np
import skimage
from etsynseg import hessian, dtvoting, utils, tracing

__all__ = [
    "GMMFixed", "fill_matched", "match_spatial_orient"
]

#=========================
# GMM
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
# match by orientation
#=========================

def fill_matched(Bseg, Oseg, B_match):
    """ fill holes of matched image
    :param Bseg, Oseg: binary image and orientation from previous segmentation
    :param B_match: matched segmented and fitted images
    :return: B_fill
        B_fill: filled matched image
    """
    # setup
    tr = tracing.Trace(Bseg, Oseg)
    # shape_yx = Bseg[0].shape
    B_fill = np.zeros(Bseg.shape, dtype=np.int_)

    # fix one slice
    def calc_one(iz):
        # get traces using depth-first scan
        yx_trs, _ = tr.dfs2d(iz)
        for yx_tr_i in yx_trs:
            # positions of pixels in the trace
            pos = tuple(np.transpose(yx_tr_i))
            # corresponding pixels in matched image
            b_match = B_match[iz][pos]
            i_match = np.nonzero(b_match)[0]
            # fill zeros inside the matched image
            if len(i_match) > 0:
                tr_filled = np.zeros(len(pos[0]), dtype=np.int_)
                tr_filled[np.min(i_match):np.max(i_match)+1] = 1
                B_fill[iz][pos] = tr_filled

    # fix all slices
    for iz in range(Bseg.shape[0]):
        calc_one(iz)

    return B_fill

def match_spatial_orient(Bseg, Oseg, Bfit, sigma_hessian, sigma_tv):
    """ filter segmented image by dO, and fill holes
    :param Bseg, Oseg: binary image and orientation from previous segmentation
    :param Bfit: binary image from moo fitting
    :param sigma_<hessian,tv>: sigma's for calculating orientation, tv on fitted surface
    :return: Bmatch
        Bmatch: matched image
    """
    # spatial closeness: TV for fit
    _, Ofit = hessian.features3d(Bfit, sigma=sigma_hessian)
    Sfit_tv, Ofit_tv = dtvoting.stick3d(Bfit, Ofit*Bfit, sigma=sigma_tv)

    # mask: belongs to B and close to Bfit
    # e^(-1/2) = e^(-r^2/(2*sigma_tv^2)) at r=sigma_tv, close to fill holes
    mask_fit = Sfit_tv > np.exp(-1/2)
    mask_fit = skimage.morphology.binary_closing(mask_fit)
    mask = (mask_fit*Bseg).astype(bool)

    # difference in orientation
    dO = utils.absdiff_orient(Oseg, Ofit_tv)

    # estimate dO threshold using GMM
    gmm = GMMFixed(means_fixed=(0, np.pi/2))
    dO_thresh = gmm.fit_threshold(dO[mask])

    # match
    # by threshold
    Bmatch = Bseg * (dO < dO_thresh) * mask

    # find connected
    # Bmatch_connect = next(iter(
    #     utils.extract_connected(Bmatch_raw, n_keep=1, connectivity=3)
    # ))[1]

    # fill holes
    Bmatch = fill_matched(Bseg, Oseg, Bmatch)

    return Bmatch

