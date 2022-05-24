""" Detect membrane-like features from the image.
"""

import numpy as np
from etsynseg import imgutil, pcdutil
from etsynseg import features, nonmaxsup, dtvoting

__all__ = [
    # detect
    "suppress_by_orient", "filter_by_value", "detect_memlike",
]


def suppress_by_orient(B, O, sigma, dO_thresh=np.pi/4):
    """ Apply a strong TV field, then suppress pixels where change in O is large.

    Args:
        B, O (np.ndarray): Input binary image and orientation, with shape=(nz,ny,nx).
        dO_thresh (float, in rad): Suppress pixels whose change in O after TV is >= this value.

    Returns:
        Bsupp (np.ndarray): Binary mask of image with 1's at non-suppressed pixels and 0's otherwise, with shape=(nz,ny,nx).
        Stv (np.ndarray): Saliency after TV, with shape=(nz,ny,nx).
    """
    # apply strong tv field
    Stv, Otv = dtvoting.stick3d(B, O, sigma)
    # calculate change in O
    dO = imgutil.orients_absdiff(Otv, O)
    # mask of pixels with small dO
    Bsupp = B*(dO < dO_thresh)
    return Bsupp, Stv

def filter_by_value(B, V, B_guide, factor):
    """ Filter out pixels by its value.

    For each slice, calculate the number of pixels in the guide (ng) and those detected (nd).
    Only keep detected pixels with values among the top ng*factor.
    
    If n_mem=2 membranes are desired, then in the ideal case the membrane pixels have the largest values,
    and the size of each membrane is approximately the size of the guide, then factor=2.
    In practice, factor can be set to e.g. 1.5*n_mem in case that some noise pixels also have large values.

    Args:
        B (np.ndarray): Binary image for pixel positions, shape=(nz,ny,nx).
        V (np.ndarray): Value for each pixel, with shape=(nz,ny,nx).
        B_guide (np.ndarray): Binary image containing the guide, with shape=(nz,ny,nx).
        factor (float): The factor for keeping pixels. Can be set to 1.5 * the number of membranes.
    
    Returns:
        B_filt (np.ndarray): Filtered pixels, shape=(nz,ny,nx).
    """
    # filter for each slice
    B_filt = np.zeros_like(B)
    for i in range(B.shape[0]):
        # the fraction of candidate pixels, clipped to [0,1]
        frac_cand = np.sum(B_guide[i])*factor/np.sum(B[i])
        frac_cand = np.clip(frac_cand, 0, 1)
        # set the threshold of values
        v_thresh = np.quantile(V[i][B[i] > 0], q=1-frac_cand)
        # filter by the threshold
        B_filt[i] = B[i]*(V[i] >= v_thresh)
    return B_filt

def detect_memlike(
        I, zyx_guide, B_bound, 
        sigma_gauss, sigma_tv,
        factor_filt=3, factor_supp=0.5, return_nofilt=False
    ):
    """ Detect membrane-like pixels from the image.

    First calculate ridge-like features using hessian.
    Then perform tensor voting (TV) for enhancement in the tangential direction.
    Filter out pixels by values after TV.
    Normal suppression to suppress short lines nearby mem-like structures.

    Args:
        I (np.ndarray): Image (bright-field, membranes appear black). Shape=(nz,ny,nx).
        guide (np.ndarray): Points in the guide surface, shape=(npts_guide,3). Assumed sorted.
        bound (np.ndarray): The bounding region, shape=I.shape.
        sigma_gauss (float): Decay lengthscale for gaussian smoothing.
            Can be set to the membrane thickness.
        sigma_tv (float): Decay lengthscale for stick tensor voting.
            Can be set to be larger than the membrane thickness but smaller than the cleft width.
        factor_filt (float): Factor for filtering out pixel by values. See filter_by_value's doc.
            Can be set to 1.5 * the number of membranes
        factor_supp (float): Factor for normal suppression.
            The decay lengthscale of the suppression field = factor_supp * avg len of guide in each slice.
        return_nofilt (bool): Whether to return the binary image before filtering.

    Returns:
        B_detect (np.ndarray): Binary image with detected pixels, shape=(nz,ny,nx).
        O_detect (np.ndarray): Orientation of the detected pixels, shape=(nz,ny,nx).
        B_nofilt (np.ndarray, optional): Return the binary image before filtering if return_nofilt=True, with shape=(nz,ny,nx).
    """
    # setup
    # negate so that membrane pixels have larger values
    Ineg = -imgutil.scale_zscore(I)
    # mask for bound, guide
    shape = Ineg.shape
    B_guide = pcdutil.points2pixels(zyx_guide, shape)
    
    # detect ridgelike features, nms
    S, O = features.ridgelike3d(Ineg, sigma=sigma_gauss)
    B = B_bound*nonmaxsup.nms3d(S, O)

    # tensor voting, nms
    # tv on B rather than S gives less scattered results
    if sigma_tv > 0:
        Stv, Otv = dtvoting.stick3d(B, B*O, sigma=sigma_tv)
        Btv = B_bound*nonmaxsup.nms3d(Stv, Otv)
    else:
        Stv = S
        Btv = B

    # filter by value
    # filter on S*Stv rather than Stv, adds info from the original image
    Btv_filt = filter_by_value(
        Btv, S*Stv, B_guide, factor=factor_filt
    )

    # recalculate orientation of the filtered pixels
    _, Oref = features.ridgelike3d(Btv_filt, sigma=sigma_gauss)

    # normal suppression
    # sigma is set to factor_supp*length of the guide
    sigma_supp = factor_supp*len(zyx_guide)/shape[0]
    Bsupp, _ = suppress_by_orient(
        Btv_filt, Oref*Btv_filt, sigma=sigma_supp
    )

    # final assignment
    B_detect = Bsupp
    O_detect = Oref * Bsupp

    if return_nofilt:
        B_nofilt = Btv
        return B_detect, O_detect, B_nofilt
    else:
        return B_detect, O_detect
