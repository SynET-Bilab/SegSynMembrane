#!/usr/bin/env python
""" hessian
"""

import numpy as np
import numba
from synseg.utils import gaussian

__all__ = [
    "hessian2d", "features2d_H1", "features2d_H2", "features3d"
]


#=========================
# hessian
#=========================

def hessian2d(I, sigma):
    """ calculate hessian
    :param sigma: if sigma=0, no gaussian smoothing
    """
    # referenced skimage.feature.hessian_matrix
    # here x,y are made explicit
    # I[y,x]
    Ig = gaussian(I, sigma)
    Hx = np.gradient(Ig, axis=1)
    Hy = np.gradient(Ig, axis=0)
    Hxx = np.gradient(Hx, axis=1)
    Hxy = np.gradient(Hx, axis=0)
    Hyy = np.gradient(Hy, axis=0)
    return Hxx, Hxy, Hyy

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

# def features3d(I, sigma, method="hessian2"):
#     """ stickness and orientation based on 2d Hessian for each slice
#     param method: hessian1 - H, hessian2 - HH 
#     return: S - saliency, O - tangent of max-amp-eigvec
#     """
#     if method == "hessian2":
#         func_hessian = features2d_H2
#     elif method == "hessian":
#         func_hessian = features2d_H1

#     S = np.zeros(I.shape, dtype=float)
#     O = np.zeros_like(S)
#     nz = S.shape[0]

#     for i in range(nz):
#         S[i], O[i] = func_hessian(I[i], sigma)

#     return S, O

@numba.njit(parallel=True)
def features3d(I, sigma):
    """ stickness and orientation based on 2d Hessian2 for each slice
    :param I: shape=(nz,ny,nx)
    :return: S - saliency, O - tangent of max-amp-eigvec
    """
    # notes on the removal of method selection:
    #   numba.njit does not support skimage.filters.gaussian
    #   numba.objmode does not support if-else
    
    # gaussian on 3d image
    with numba.objmode(Ig="float64[:,:,:]"):
        Ig = gaussian(I.astype(np.float64), sigma)

    # create arrays
    S = np.zeros(Ig.shape, dtype=np.float64)
    O = np.zeros(Ig.shape, dtype=np.float64)
    nz = S.shape[0]

    for i in numba.prange(nz):
        with numba.objmode(S_i="float64[:,:]", O_i="float64[:,:]"):
            # note sigma=0 because 3d gaussian is already applied
            S_i, O_i = features2d_H2(Ig[i], 0)
        S[i] = S_i
        O[i] = O_i

    return S, O
