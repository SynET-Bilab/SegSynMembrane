""" refine
usage:
    dO, mask = diff_fit_seg(mootools, indiv, O_seg, sigma_gauss, sigma_tv)
    B_filt = filter_seg(mootools.B, dO, mask, factor)
"""
import numpy as np
from synseg import hessian
from synseg import dtvoting
from synseg import utils

__all__ = [
    "diff_fit_seg",
    "gaussian_pdf", "mixture_gaussian",
    "filter_seg"
]

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
    # e^(-1/2) = e^(-r^2/(2*sigma_tv^2)) at r=sigma_tv
    mask = mootools.B * (Sfit_tv > np.exp(-1/2))
    mask = mask.astype(bool)

    # difference in orientation
    dO = utils.absdiff_orient(O_seg, Ofit_tv)
    dO = mask*dO
    return dO, mask

def gaussian_pdf(x2, sigma):
    """ gaussian pdf function
    :param x2: x^2
    :param sigma: sigma for gaussian
    """
    return np.exp(-x2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)


def mixture_gaussian(data, tol=1e-4, max_iter=100, sigma_init=0.1, data_span=np.pi/2):
    """ estimate gaussian peak from data using EM algorithm
    :param data: 1d array of data
    :param tol, max_iter: convergence criteria
    :param sigma_init: initial sigma, set to ~0 for finding peak around 0
    :param data_span: range of data = (0, data_span)
    :return: sigma
        sigma: estimated sigma after EM
    """
    # setup frequently used values
    p_cond_uniform = 1/data_span
    data2 = data**2

    # iterative E-M steps
    sigma_curr = sigma_init
    for _ in range(max_iter):
        # E-step: posterior given sigma
        p_cond_gauss = gaussian_pdf(data2, sigma_curr)
        p_post_gauss = p_cond_gauss / (p_cond_gauss + p_cond_uniform)

        # M-step: max likelihood by setting sigma to avg
        sigma_next2 = np.sum(p_post_gauss*data2)/np.sum(p_post_gauss)
        sigma_next = (sigma_next2)**0.5

        # termination
        if np.abs(sigma_next-sigma_curr) <= tol:
            break

        sigma_curr = sigma_next

    return sigma_next

def filter_seg(B_seg, dO, mask, factor=2):
    """ filter segmented image by dO
    :param B_seg: binary image from prev segmentation
    :param dO, mask: results from diff_fit_seg()
    :param factor: dO_threshold = factor*dO_sigma, dO_sigma is from mixture_gaussian()
    :return: B_filt
        B_filt: filtered B_seg
    """
    # estimate peak
    dO_sigma = mixture_gaussian(dO[mask])
    # filter
    dO_thresh = factor * dO_sigma
    B_filt = B_seg * (dO < dO_thresh) * mask
    return B_filt
