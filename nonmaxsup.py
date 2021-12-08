#!/usr/bin/env python
""" nonmaxsup: non-maximum suppression
"""

import numpy as np
import numba
from synseg.hessian import gaussian

__all__ = [
    "nms2d", "nms3d"
]

#=========================
# find local maximum
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
    :param local_max: bool array, shape=S.shape
    :param N_range: (N_min, N_max), in rad, in (0, pi)
    :param S: values of saliency, shape=(ny, nx)
    :param N: values of normal direction, in (0, pi)
    :param mask: bool array, mask according to other criteria, shape=S.shape
    :return: None, local_max[pts] is filled
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


#=========================
# non-max suppression
#=========================

def nms2d(S, O, sigma=1, qthreshold=0.5):
    """ non-maximum suppresion of S[ny, nx]
    :param S: saliency
    :param O: orientation (tangent)
    :param sigma: gaussian smoothing
    :param qthreshold: threshold on quantile of S
    :return: local_max (int array, for easier multiplication between masks)
    """
    # global masks
    mask_boundary = set_mask_boundary(S)
    mask_S = S > np.quantile(S, qthreshold)
    mask = mask_boundary & mask_S

    # smoothing
    if sigma > 0:
        S = gaussian(S, sigma)

    # normal direction, mod pi
    N = np.mod(O+np.pi/2, np.pi)

    # local max in four directions
    local_max = np.zeros(S.shape, dtype=bool)
    kwargs = dict(S=S, N=N, mask=mask)
    set_local_max(local_max, N_range=(0, np.pi/4), **kwargs)
    set_local_max(local_max, N_range=(np.pi/4, np.pi/2), **kwargs)
    set_local_max(local_max, N_range=(np.pi/2, np.pi*3/4), **kwargs)
    set_local_max(local_max, N_range=(np.pi*3/4, np.pi), **kwargs)

    # return int-type array
    return local_max.astype(np.int64)


def nms3d_python(S, O, qthreshold=0.5):
    """ non-maximum suppresion of S[nz, ny, nx], python version
    :param S: assumed already gaussian-smoothed
    """
    # find local max for each slice
    local_max = np.zeros(S.shape, dtype=np.int64)
    nz = S.shape[0]
    for i in range(nz):
        local_max[i] = nms2d(
            S[i], O[i],
            sigma=0,  # set to 0 because gaussian is already done
            qthreshold=qthreshold
        )
    return local_max


@numba.njit(parallel=True)
def nms3d_numba(S, O, qthreshold=0.5):
    """ non-maximum suppresion of S[nz, ny, nx], numba version
    :param S: assumed already gaussian-smoothed
    """
    # find local max for each slice
    local_max = np.zeros(S.shape, dtype=np.int64)
    nz = S.shape[0]
    for i in numba.prange(nz):
        with numba.objmode(local_max_i="int64[:,:]"):
            local_max_i = nms2d(
                S[i], O[i],
                sigma=0,  # set to 0 because gaussian is already done
                qthreshold=qthreshold
            )
        local_max[i] = local_max_i
    return local_max


def nms3d(S, O, sigma=1, qthreshold=0.5, method="numba"):
    """ non-maximum suppresion of S[nz, ny, nx], slice by slice
    :param S: saliency
    :param O: orientation (tangent)
    :param sigma: gaussian smoothing
    :param qthreshold: threshold on quantile of S
    :param method: python or numba
    :return: local_max (int array, for easier multiplication between masks)
    """
    # 3d gaussian smoothing
    Sg = gaussian(S, sigma)
    # perform nonmaxsup
    if method == "numba":
        return nms3d_numba(Sg, O, qthreshold=qthreshold)
    elif method == "python":
        return nms3d_python(Sg, O, qthreshold=qthreshold)
    else:
        raise ValueError("method: python or numba")
