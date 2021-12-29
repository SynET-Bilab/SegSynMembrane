#!/usr/bin/env python
""" dtvoting: for steerable tensor voting

SteerTV:
    notation c.f. Franken2006
    wm = exp(-r**2/(2*sigma**2))*((x+iy)/r)**m
    cm = S*exp(-i*m*O)

    stick field: S=|U|, O=arg(U)/2
    U = w0*c2c + 4*w2*c0 + 6*w4*c2 + 4*w6*c4 + w8*c6
    *: convolution, A*B = ifft(fft(A)*fft(B))

    ball field:
    B = Re[6*w0*c0 + 8*w2*c2 + 2*w4*c4]
"""

import numpy as np
from numpy import fft
import numba
import pandas as pd
from synseg import utils

__all__ = [
    # stick tv
    "prep_wmfft_stick", "stick2d_wmfft", "stick2d", "stick3d",
    # ball tv
    "prep_wmfft_ball", "ball2d_wmfft", "ball2d", "ball3d",
    # suppression
    "suppress_by_orient"
]


#=========================
# prepare wmfft general
#=========================

def prep_wmfft(ny, nx, sigma, m, excluded_r2, origin, n_terms):
    """ prepare wmfft, radial function f(r)=r^m e^(-r^2/2sigma^2)
    :param ny, nx: ny=nrow, nx=ncol
    :param sigma, m: parameters of radial function
    :param excluded_r2: set r2<excluded_r2 to zero
    :param origin: value at origin, to avoid nan
    :param n_terms: take the first n terms in wmfft, 5 for stick field, 3 for ball field
    :return: wmfft
        wmfft: fft of [w0,w2,w4,w6,w8][:n_terms], shape of wi=(ny,nx)
    """
    # spatial coords
    # col-x, row-y, in image convention
    # fftfreq is used to easily construct x,y in correct order
    x = fft.fftfreq(nx, d=1/nx).reshape((1, nx))
    y = fft.fftfreq(ny, d=1/ny).reshape((ny, 1))

    # radial terms, scaled so that max=1
    r2 = x**2 + y**2
    term_exp_max = (m**0.5*sigma)**m*np.exp(-m/2)
    # divide by 16, the scale factor for orientation (Franken2006)
    term_exp = r2**(m/2)*np.exp(-r2/(2*sigma**2)) / term_exp_max / 16
    term_exp[r2<excluded_r2] = 0

    # fraction term
    term_frac2 = (x+1j*y)**2/r2
    # set origin to avoid nan
    term_frac2[0, 0] = origin

    # calculate wm accumulatively
    wmfft = np.zeros((n_terms, ny, nx), dtype=np.complex128)
    wm = term_exp
    wmfft[0] = fft.fft2(wm)
    for i in range(1, n_terms):
        wm = wm*term_frac2
        wmfft[i] = fft.fft2(wm)

    return wmfft


#=========================
# stick field
#=========================

def prep_wmfft_stick(ny, nx, sigma):
    """ prepare wmfft for stick field, e^(-r^2/2sigma^2), origin=1
    excluded_r2 is set to 1, due to some steertv issues at r=0
    :param ny, nx: ny=nrow, nx=ncol
    :return: wmfft=[w0,w2,w4,w6,w8], shape=(5,ny,nx)
    """
    wmfft = prep_wmfft(ny, nx, sigma,
        m=0, excluded_r2=1, origin=1, n_terms=5
    )
    return wmfft

def stick2d_wmfft(S, O, wmfft):
    """ ball field tv for 2d slice
    :param S, O: shape=(ny,nx)
    :param wmfft: fft of [w0,w2,w4,w6,w8]
    :return: S_tv, O_tv
    """
    # orientation term
    term_exp2 = np.exp(-2j*O)
    c0 = S
    c2 = c0*term_exp2
    c4 = c2*term_exp2
    c6 = c4*term_exp2
    c2c = np.conj(c2)
    c_arr = [c2c, c0, c2, c4, c6]

    # coef
    coef = [1, 4, 6, 4, 1]

    # sum the series
    # U = w0*c2c + 4*w2*c0 + 6*w4*c2 + 4*w6*c4 + w8*c6
    series = [
        coef[m]*fft.ifft2(wmfft[m]*fft.fft2(c_arr[m]))
        for m in range(5)
    ]
    U = np.sum(series, axis=0)

    # calculate saliency and orientation
    S_tv = np.abs(U)
    O_tv = 0.5*np.angle(U)
    return S_tv, O_tv

def stick2d(S, O, sigma):
    """ tv for 2d slice, e^(-r^2/2sigma^2), origin=1
    :param S, O: shape=(ny,nx)
    :param sigma: scale in pixel
    :return: S_tv, O_tv
    """
    ny, nx = S.shape
    wmfft = prep_wmfft_stick(ny, nx, sigma)
    S_tv, O_tv = stick2d_wmfft(S, O, wmfft)
    return S_tv, O_tv

@numba.njit(parallel=True)
def stick3d(S, O, sigma):
    """ tv for 2d stack, using numba parallel
    :return: S_tv, O_tv
    """
    nz, ny, nx = S.shape
    with numba.objmode(wmfft='complex128[:,:,:]'):
        wmfft = prep_wmfft_stick(ny=ny, nx=nx, sigma=sigma)


    # calc S, O for each slice
    S_tv = np.zeros(S.shape, dtype=np.float64)
    O_tv = np.zeros(O.shape, dtype=np.float64)
    for i in numba.prange(nz):
        with numba.objmode(S_tv_i='float64[:,:]', O_tv_i='float64[:,:]'):
            S_tv_i, O_tv_i = stick2d_wmfft(S[i], O[i], wmfft)
        S_tv[i] = S_tv_i
        O_tv[i] = O_tv_i
    return S_tv, O_tv


#=========================
# ball field
#=========================

def prep_wmfft_ball(ny, nx, sigma):
    """ prepare wmfft for ball field, e^(-r^2/2sigma^2), origin=0
    excluded_r2 is set to 3, to avoid normalsup of membrane pixels
    :param ny, nx: ny=nrow, nx=ncol
    :return: wmfft
        wmfft: fft of [w0,w2,w4], shape=(3,ny,nx)
    """
    wmfft = prep_wmfft(ny, nx, sigma,
        m=0, excluded_r2=3, origin=0, n_terms=3
    )
    return wmfft

def ball2d_wmfft(S, O, wmfft):
    """ ball field tv for 2d slice, r^2e^(-r^2/2sigma^2), origin=0
    :param S, O: shape=(ny,nx)
    :param wmfft: fft of [w0,w2,w4,w6,w8]
    :return: S_tv, O_tv
    """
    # orientation term
    term_exp2 = np.exp(-2j*O)
    c0 = S
    c2 = c0*term_exp2
    c4 = c2*term_exp2
    c_arr = [c0, c2, c4]

    # coef
    coef = [6, 8, 2]

    # sum the series
    # U = Re(6*w0*c0 + 8*w2*c2 + 2*w4*c4)
    series = [np.real(
        coef[m]*fft.ifft2(wmfft[m]*fft.fft2(c_arr[m]))
    ) for m in range(3)]
    S = np.sum(series, axis=0)

    return S

def ball2d(S, O, sigma):
    """ ball field tv for 2d slice
    :param S, O: shape=(ny,nx)
    :param sigma: scale in pixel
    :return: S_tv, O_tv
    """
    ny, nx = S.shape
    wmfft = prep_wmfft_ball(ny, nx, sigma)
    S_tv = ball2d_wmfft(S, O, wmfft)
    return S_tv

@numba.njit(parallel=True)
def ball3d(S, O, sigma):
    """ ball field tv for a stack of 2d's
    :return: S_tv
    """
    nz, ny, nx = S.shape
    with numba.objmode(wmfft='complex128[:,:,:]'):
        wmfft = prep_wmfft_ball(ny=ny, nx=nx, sigma=sigma)
    # calc S, O for each slice
    S_tv = np.zeros(S.shape, dtype=np.float64)
    for i in numba.prange(nz):
        with numba.objmode(S_tv_i='float64[:,:]'):
            S_tv_i = ball2d_wmfft(S[i], O[i], wmfft)
        S_tv[i] = S_tv_i
    return S_tv


#=========================
# suppress by orientation
#=========================

def suppress_by_orient(nms, O, sigma, dO_threshold=np.pi/4):
    """ apply strong tv field, suppress pixels where change in O is large
    :param nms, O: shape=(nz,ny,nx)
    :param dO_threshold: threshold of change in O
    :return: supp, Stv
        supp: binary mask of image after suppression
        Stv: Stv after applying strong field
    """
    # apply strong tv field
    Stv, Otv = stick3d(nms, O, sigma)
    # calculate change in O
    dO = utils.absdiff_orient(Otv, O)
    # mask of pixels with small dO
    supp = nms*(dO<dO_threshold)
    return supp, Stv


#=========================
# deprecated
#=========================

# def stats_by_seg(L, O, sigma, stats=np.sum):
#     """ apply tv, order by stats (e.g. sum)
#     :param L, O: 2d label, orientation
#     :param sigma: sigma for sticktv, e.g. 2*cleft
#     :param stats: function to calculate stats
#     :return: df["label", "count", "S_stats"]
#         df: sorted by S_stats, descending
#     """
#     # tv on binary image
#     nms = (L > 0).astype(np.int_)
#     Stv, _ = stick2d(nms, O, sigma)

#     # stat for each label
#     columns = ["label", "count", "S_stats"]
#     data = []
#     for l in np.unique(L[L > 0]):
#         pos_l = np.nonzero(L == l)
#         Stv_l = Stv[pos_l]
#         data_l = [
#             l, len(pos_l[0]), stats(Stv_l)
#         ]
#         data.append(data_l)

#     # make dataframe, sort
#     df = pd.DataFrame(data=data, columns=columns)
#     df = (df.sort_values("S_stats", ascending=False)
#           .reset_index(drop=True)
#           )
#     df = df.astype({f: int for f in ["label", "count"]})

#     return df
