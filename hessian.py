#!/usr/bin/env python
""" hessian
"""

import numpy as np
import numba
from itertools import combinations_with_replacement
from synseg.utils import gaussian, reverse_coord

__all__ = [
    # 2d hessian, saliency
    "hessian2d", "features2d_H1", "features2d_H2", "features3d",
    # 3d hessian, surface normal
    "hessian3d", "symmetric_image", "surface_norm"
]


#=========================
# hessian
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

#=========================
# surface normal
#=========================

def hessian3d(I, sigma):
    """ calculate hessian 3d
    :param I: image, shape=(nz,ny,nx)
    :param sigma: gaussian smoothing, no smooth if sigma=0
    :return: Hxx, Hxy, Hxz, Hyy, Hyz, Hzz
    """
    # referenced skimage.feature.hessian_matrix
    # here x,y,z are made explicit
    # I[z,y,x]
    Ig = gaussian(I, sigma)
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

def symmetric_image(S_elems):
    """ make symmetric image from elements
    ref: skimage.feature.corner._symmetric_image
    :param S_elems: e.g. Hxx, Hxy, Hyy
    :return: H_mats
        H_mats: H_mats[z,y,x] = hessian matrix at (z,y,x)
    """
    image = S_elems[0]
    symmetric_image = np.zeros(
        image.shape + (image.ndim, image.ndim),
        dtype=S_elems[0].dtype
    )
    for idx, (row, col) in enumerate(
        combinations_with_replacement(range(image.ndim), 2)
    ):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image

def surface_norm_one(H_mat, del_ref):
    """ calculate surface normal at one location, via hessian
    :param H_mat: 3*3 hessian matrix at the location
    :param del_ref: [x,y,z] of location - [x,y,z] of reference
    :return: evec_out
        evec_out: outward normal vector, [x,y,z]
    """
    evals, evecs = np.linalg.eigh(H_mat)
    # max eig by abs
    imax = np.argmax(np.abs(evals))
    evec = evecs[:, imax]
    # align direction with outward
    evec_out = evec if np.dot(evec, del_ref) >= 0 else -evec
    return evec_out

@numba.njit(parallel=True)
def surface_norm(I, xy_ref, sigma=1):
    """ calculate surface normal at nonzeros of I
    :param I: image, shape=(nz,ny,nx)
    :param xy_ref: [x,y] of a reference point inside
    :param sigma: gaussian smoothing
    :return: xyz, norm
        xyz: coordinates, [[x1,y1,z1],...]
        norm: normal vectors, [[nx1,ny1,nz1],...]
    """
    # nonzero pixels
    pos = np.nonzero(I)
    n_pts = len(pos[0])
    # coordinates and hessian
    with numba.objmode(xyz="int64[:,:]", H_mats="float64[:,:,:]"):
        xyz = reverse_coord(np.transpose(pos))
        H_mats = symmetric_image(hessian3d(I, sigma))[pos]
    # normal
    norm = np.zeros((n_pts, 3))
    for i in numba.prange(n_pts):
        with numba.objmode(norm_i="float64[:]"):
            norm_i = surface_norm_one(
                H_mats[i],
                del_ref=xyz[i]-np.array(
                    [xy_ref[0], xy_ref[1], xyz[i][-1]]
                )
            )
        norm[i] = norm_i
    return xyz, norm
