#!/usr/bin/env python
""" normalsup: suppression field in the normal direction
"""

import numpy as np
from numpy import fft
import numba

__all__ = [
    "nsfield2d", "nsfield3d"
]

def nsfield_kernel(ny, nx, sigma, mask_direction, excluded_r2=3):
    """ kernel of suppression field along given direction
    :param ny, nx: dimensions
    :param sigma: decay lengthscale as in exp(-r^2/(2*sigma^2))
    :param mask_direction: mask for direction, e.g. lambda x,y: x==y
    :param excluded_r2: set r^2 < excluded_r2 to 0
    :return: kernel[ny,nx]
    """
    # setup x, y as fftfreq
    # set type to int, so that comparisons like x==y work as expected
    x = fft.fftfreq(nx, d=1/nx).reshape((1, nx)).astype(np.int_)
    y = fft.fftfreq(ny, d=1/ny).reshape((ny, 1)).astype(np.int_)
    # exp decay
    r2 = x**2 + y**2
    fr = np.exp(-r2/(2*sigma**2))
    # exclude a region surrounding the origin
    fr[r2 < excluded_r2] = 0.
    # mask to obtain certain direction
    mask = mask_direction(x, y)
    kernel = fr * mask
    return kernel

def nsfield2d(S, O, sigma, excluded_r2=3):
    """ normal suppression field in 2d
    :param S: saliency
    :param O: orientation (tangent)
    :param sigma: decay lengthscale as in exp(-r^2/(2*sigma^2))
    :param excluded_r2: set r^2 < excluded_r2 to 0
    :return: S_supp[ny,nx]
    """
    # mask for each range of directions
    directions = [
        dict(N_range=(0, np.pi/8), mask=(lambda x, y: y == 0)),
        dict(N_range=(np.pi/8, np.pi*3/8), mask=(lambda x, y: x == y)),
        dict(N_range=(np.pi*3/8, np.pi*5/8), mask=(lambda x, y: x == 0)),
        dict(N_range=(np.pi*5/8, np.pi*7/8), mask=(lambda x, y: x == -y)),
        dict(N_range=(np.pi*7/8, np.pi), mask=(lambda x, y: y == 0))
    ]
    # normal direction, mod pi
    N = np.mod(O+np.pi/2, np.pi)

    # calculate suppression field for all directions
    S_supp = np.zeros(S.shape, dtype=np.float64)
    for d in directions:
        # select points in the direction range
        pts = (N >= d["N_range"][0]) & (N < d["N_range"][1])
        S_masked_d = S*pts
        # get field kernel
        supp_d = nsfield_kernel(*S.shape, sigma, 
            mask_direction=d["mask"], excluded_r2=excluded_r2
        )
        # convolve
        S_supp_d = fft.ifft2(
            fft.fft2(S_masked_d)*fft.fft2(supp_d)
        ).real
        # accumulate
        S_supp += S_supp_d
    return S_supp

@numba.njit(parallel=True)
def nsfield3d(S, O, sigma, excluded_r2=3):
    """ normal suppression field in 3d
    :param S: saliency
    :param O: orientation (tangent)
    :param sigma: decay lengthscale as in exp(-r^2/(2*sigma^2))
    :param excluded_r2: set r^2 < excluded_r2 to 0
    :return: S_supp[nz,ny,nx]
    """
    # find local max for each slice
    S_supp = np.zeros(S.shape, dtype=np.int64)
    nz = S.shape[0]
    for i in numba.prange(nz):
        with numba.objmode(S_supp_i="float64[:,:]"):
            S_supp_i = nsfield2d(
                S[i], O[i], sigma, excluded_r2
            )
        S_supp[i] = S_supp_i
    return S_supp
