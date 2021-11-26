#!/usr/bin/env python
""" imgprocess: for image processing
"""

import numpy as np
import skimage
import skimage.feature

#=========================
# basics
#=========================

def zscore(img):
    """ return zscore of img """
    I = np.array(img)
    z = (I-I.mean())/I.std()
    return z

def negate(img):
    """ switch foreground between white and dark
    zscore then negate
    """
    I = np.array(img)
    std = I.std()
    if std > 0:
        return -(I-I.mean())/std
    else:
        return np.zeros_like(I)


#=========================
# hessian
#=========================

def hessian_matrix(I, sigma):
    """ calculate hessian
    """
    # referenced skimage.feature.hessian_matrix
    # here x,y are made explicit
    Ig = skimage.filters.gaussian(I, sigma=sigma, mode="nearest")
    Hx = np.gradient(Ig, axis=1)
    Hy = np.gradient(Ig, axis=0)
    Hxx = np.gradient(Hx, axis=1)
    Hxy = np.gradient(Hx, axis=0)
    Hyy = np.gradient(Hy, axis=0)
    return Hxx, Hxy, Hyy

def features_hessian(I, sigma):
    """ stickness and orientation based on 2d Hesssian
    return: S - saliency, O - tangent of max-amp-eigvec
    """
    # hessian 2d
    Hxx, Hxy, Hyy = hessian_matrix(I, sigma)

    # eigenvalues: l2+, l2-
    # eigenvectors: e2+, e2-
    # mask: select edge-like where l- > |l+|
    mask_tr = (Hxx+Hyy) < 0
    # saliency: |l+ - l-|
    S = mask_tr*np.sqrt((Hxx-Hyy)**2 + 4*Hxy**2)
    # orientation: e- (normal) -> pi/2 rotation (tangent) -> e+
    O = mask_tr*0.5*np.angle(Hxx-Hyy+2j*Hxy)
    return S, O

def features_hessian2(I, sigma):
    """ stickness and orientation based on 2d Hesssian^2
    return: S2 - saliency, O2 - tangent of max-amp-eigvec
    """
    # hessian 2d
    Hxx, Hxy, Hyy = hessian_matrix(I, sigma)
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


#=========================
# non-maximum suppression
#=========================

def set_mask_boundary(I):
    """ mask out grids on the boundary """
    mask_boundary = np.ones(I.shape, dtype=bool)
    for i in [0, -1]:
        mask_boundary[i, :] = 0  # y=0,ny
        mask_boundary[:, i] = 0  # x=0,nx
    return mask_boundary

def values_xminus(I, pts):
    """ values of grids that are xminus to I[pts] """
    # pts[:, 1:] - new loc = (y, x-1)
    # I[:, :-1] - match shape of pts
    return I[:, :-1][pts[:, 1:]]

def values_xplus(I, pts):
    """ values of grids that are xplus to I[pts] """
    # I[:, 1:] - new loc = (y, x+1)
    # pts[:, :-1] - match shape of I
    return I[:, 1:][pts[:, :-1]]

def values_yminus(I, pts):
    """ values of grids that are yminus to I[pts] """
    # pts[1:, :] - new loc = (y-1, x)
    # I[:-1, :] - match shape of pts
    return I[:-1, :][pts[1:, :]]

def values_yplus(I, pts):
    """ values of grids that are yplus to I[pts] """
    # I[1:, :] - new loc = (y+1, x)
    # pts[:-1, :] - match shape of I
    return I[1:, :][pts[:-1, :]]

def set_local_max(local_max, N_range, S, N, mask):
    """ set local_max of pts with N in N_range
    param local_max: bool array, shape=S.shape
    param N_range: (N_min, N_max), in rad, in (0, pi)
    param S: values of saliency, shape=(ny, nx)
    param N: values of normal direction, in (0, pi)
    param mask: bool array, mask according to other criteria, shape=S.shape
    action: fill local_max[pts]
    """
    # pts in N_range
    pts = mask & (N >= N_range[0]) & (N < N_range[1])

    # weights
    weight_x = np.cos(N[pts])**2
    weight_y = np.sin(N[pts])**2

    # values in plus/minus directions
    N_mid = np.mean(N_range)
    # x - direction
    if np.cos(N_mid) >= 0:  # N in (0, pi/2), Nplus_x = xplus
        S_Nplus_x = values_xplus(S, pts)
        S_Nminus_x = values_xminus(S, pts)
    else:  # N in (pi/2, pi), Nplus_x = xminus
        S_Nplus_x = values_xminus(S, pts)
        S_Nminus_x = values_xplus(S, pts)
    # y - direction
    # since N in (0, pi), Nplus_y = yplus
    S_Nplus_y = values_yplus(S, pts)
    S_Nminus_y = values_yminus(S, pts)

    # compare S with S_Nplus/minus
    gt_plus = S[pts] > (weight_x*S_Nplus_x + weight_y*S_Nplus_y)
    gt_minus = S[pts] > (weight_x*S_Nminus_x + weight_y*S_Nminus_y)

    # set local max
    local_max[pts] = gt_plus & gt_minus

def nonmaxsup(S, O, qthreshold=0.5):
    """ non-maximum suppresion of S
    param S: saliency
    param O: orientation (tangent)
    param qthreshold: only consider S > quantile(S, qthreshold)
    return: local_max (bool array, shape=S.shapess)
    """
    # notations: img[y, x]

    # global masks
    mask_boundary = set_mask_boundary(S)
    mask_S = S > np.quantile(S, qthreshold)
    mask = mask_boundary & mask_S

    # normal direction, mod pi
    N = np.mod(O+np.pi/2, np.pi)

    # local max in four directions
    local_max = np.zeros(S.shape, dtype=bool)
    kwargs = dict(S=S, N=N, mask=mask)
    set_local_max(local_max, N_range=(0, np.pi/4), **kwargs)
    set_local_max(local_max, N_range=(np.pi/4, np.pi/2), **kwargs)
    set_local_max(local_max, N_range=(np.pi/2, np.pi*3/4), **kwargs)
    set_local_max(local_max, N_range=(np.pi*3/4, np.pi), **kwargs)

    return local_max
