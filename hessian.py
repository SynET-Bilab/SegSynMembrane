""" hessian
"""

import numpy as np
import multiprocessing.dummy
import itertools
from etsynseg import utils

__all__ = [
    # hessians
    "hessian2d", "hessian3d", "symmetric_image",
    # saliency
    "features2d_H1", "features2d_H2", "features3d",
]

#=========================
# hessians
#=========================

def hessian2d(I, sigma):
    """ calculate hessian 2d
    :param I: image, shape=(ny,nx)
    :param sigma: gaussian smoothing, no smooth if sigma=0
    :return: Hxx, Hxy, Hyy
    """
    # referenced skimage.feature.hessian_matrix
    # here x,y are made explicit
    # I[y,x]
    Ig = utils.gaussian(I, sigma)
    Hx = np.gradient(Ig, axis=1)
    Hy = np.gradient(Ig, axis=0)
    Hxx = np.gradient(Hx, axis=1)
    Hxy = np.gradient(Hx, axis=0)
    Hyy = np.gradient(Hy, axis=0)
    return Hxx, Hxy, Hyy

def hessian3d(I, sigma):
    """ calculate hessian 3d
    :param I: image, shape=(nz,ny,nx)
    :param sigma: gaussian smoothing, no smooth if sigma=0
    :return: Hxx, Hxy, Hxz, Hyy, Hyz, Hzz
    """
    # referenced skimage.feature.hessian_matrix
    # here x,y,z are made explicit
    # I[z,y,x]
    Ig = utils.gaussian(I, sigma)
    Hx = np.gradient(Ig, axis=2)
    Hy = np.gradient(Ig, axis=1)
    Hz = np.gradient(Ig, axis=0)
    Hxx = np.gradient(Hx, axis=2)
    Hxy = np.gradient(Hx, axis=1)
    Hxz = np.gradient(Hx, axis=0)
    Hyy = np.gradient(Hy, axis=1)
    Hyz = np.gradient(Hy, axis=0)
    Hzz = np.gradient(Hz, axis=0)
    return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz

def symmetric_image(H_elems):
    """ make symmetric image from elements
    ref: skimage.feature.corner._symmetric_image
    :param H_elems: e.g. Hxx, Hxy, Hyy
    :return: H_mats
        H_mats: H_mats[z,y,x] = hessian matrix at (z,y,x)
    """
    image = H_elems[0]
    H_mats = np.zeros(
        image.shape + (image.ndim, image.ndim),
        dtype=image.dtype
    )
    iter_rc = itertools.combinations_with_replacement(
        range(image.ndim), 2)
    for idx, (row, col) in enumerate(iter_rc):
        H_mats[..., row, col] = H_elems[idx]
        H_mats[..., col, row] = H_elems[idx]
    return H_mats


#=========================
# features
#=========================

def features2d_H1(I, sigma):
    """ stickness and orientation based on 2d Hesssian
    :return: S - saliency, O - tangent of max-amp-eigvec
    """
    # hessian 2d
    Hxx, Hxy, Hyy = hessian2d(I, sigma)

    # eigenvalues: l2+, l2-
    # eigenvectors: e2+, e2-
    # mask: select edge-like where l- > |l+|
    mask_tr = (Hxx+Hyy) < 0
    # saliency: |l+ - l-|
    S = mask_tr*np.sqrt((Hxx-Hyy)**2 + 4*Hxy**2)
    # orientation: e- (normal) -> pi/2 rotation (tangent) -> e+
    O = mask_tr*0.5*np.angle(Hxx-Hyy+2j*Hxy)
    return S, O

def features2d_H2(I, sigma):
    """ stickness and orientation based on 2d Hesssian^2
    :return: S2 - saliency, O2 - tangent of max-amp-eigvec
    """
    # hessian 2d
    Hxx, Hxy, Hyy = hessian2d(I, sigma)
    # hessian*hessian
    H2xx = Hxx*Hxx + Hxy*Hxy
    H2xy = (Hxx+Hyy)*Hxy
    H2yy = Hyy*Hyy + Hxy*Hxy

    # eigenvalues: l2+, l2-
    # eigenvectors: e2+, e2-
    # mask: select edge-like where l- > |l+|
    mask_tr = (Hxx+Hyy) < 0
    # saliency: l2+ - l2- = l+^2 - l-^2
    S2 = mask_tr*np.sqrt((H2xx-H2yy)**2 + 4*H2xy**2)
    # orientation: e2+ (normal) -> pi/2 rotation (tangent) -> e2-
    O2 = mask_tr*0.5*np.angle(-H2xx+H2yy-2j*H2xy)
    return S2, O2

def features3d(I, sigma):
    """ stickness and orientation based on 2d Hessian2 for each slice
    :param I: shape=(nz,ny,nx)
    :return: S - saliency, O - tangent of max-amp-eigvec
    """
    # gaussian on 3d image
    Ig = utils.gaussian(I.astype(np.float_), sigma)

    # create arrays
    S = np.zeros(Ig.shape, dtype=np.float_)
    O = np.zeros(Ig.shape, dtype=np.float_)
    nz = S.shape[0]

    # parallel computing
    def calc_one(i):
        S[i], O[i] = features2d_H2(Ig[i], 0)
    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(nz))
    pool.close()
    return S, O
