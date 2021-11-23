#!/usr/bin/env python
""" dtvoting

Tensor voting:
    km[m] = exp(-2j*(m-1)*alpha)
    Um2 = sum([ifft2(fft2(km[m]*S)*vmfft[m] for m in range(5)])
    S_tv = abs(Um2)
    alpha_tv = 0.5*angle(Um2)

Methods:
    precalc_vmfft(nrow, ncol, sigma) -> vmfft
    tv2d(S, alpha, vmfft) -> S_tv, alpha_tv
"""

import numpy as np
import numpy.fft as fft
import multiprocessing
import functools

def precalc_vmfft(nrow, ncol, sigma):
    """ precalculate vmfft
    param nrow, ncol: col-x, row-y
    return: vmfft, shape=(5,nrow,ncol)
    """
    # spatial coords: col-x, row-y, in image convention
    # fftfreq is used to easily construct x/y in correct order
    cc = fft.fftfreq(ncol, d=1/ncol).reshape((1, ncol))
    rr = fft.fftfreq(nrow, d=1/nrow).reshape((nrow, 1))

    # spatial terms
    rc2 = cc**2 + rr**2
    term_exp = np.exp(-rc2/(2*sigma*sigma))
    term_frac = (cc+1j*rr)/np.sqrt(rc2)
    term_frac[0, 0] = 1.  # set the term at origin to 1

    # prefactor gamma
    gamma_m = np.array([1, 4, 6, 4, 1])

    # calc vmfft
    vmfft = np.array([
        gamma_m[m]*fft.fft2(term_exp*term_frac**(2*m))
        for m in range(5)
    ])
    return vmfft

def tv2d(S, alpha, vmfft):
    """ tv for 2d slice
    param S, alpha: shape=(ny,nx)
    param vmfft: array of 5 elements, each with shape=(ny,nx)
    return: S_tv, alpha_tv
    """
    # orientation term
    km_234 = [np.exp(-2j*(m-1)*alpha) for m in [2, 3, 4]]
    km_01 = [np.conj(km_234[0]), 1]
    km = km_01 + km_234
    
    # sum the series
    series = [
        fft.ifft2(fft.fft2(km[m]*S)*vmfft[m])
        for m in range(5)
    ]
    Um2 = np.sum(series, axis=0)

    # calculate saliency and orientation
    S_tv = np.abs(Um2)
    alpha_tv = 0.5*np.angle(Um2)
    return S_tv, alpha_tv

def tv2d_slice(S, alpha, vmfft, i):
    """ tv2d on slice i for the 3d inputs(S, alpha)
    param S, alpha: shape=(nz,ny,nx)
    """
    return tv2d(S[i], alpha[i], vmfft)

def tv3d(S, alpha, sigma):
    """ tv for 3d volume
    """
    # precalc vmfft
    nz, ny, nx = S.shape
    vmfft = precalc_vmfft(nrow=ny, ncol=nx, sigma=sigma)

    # calc S, alpha for each slice
    S_tv = np.zeros_like(S)
    alpha_tv = np.zeros_like(alpha)
    for i in range(nz):
        S_tv[i], alpha_tv[i] = tv2d(S[i], alpha[i], vmfft)
    return S_tv, alpha_tv

def tv3d_mp(S, alpha, sigma):
    """ tv for 3d volume, multiprocessing
    """
    # precalc vmfft
    nz, ny, nx = S.shape
    vmfft = precalc_vmfft(nrow=ny, ncol=nx, sigma=sigma)

    # calc S, alpha for each slice
    tv2d_partial = functools.partial(tv2d_slice, S, alpha, vmfft)
    p = multiprocessing.Pool()
    result = p.map(tv2d_partial, list(range(nz)))

    # assemble result
    S_tv = np.zeros_like(S)
    alpha_tv = np.zeros_like(alpha)
    for i in range(nz):
        S_tv[i] = result[i][0]
        alpha_tv[i] = result[i][1]
    return S_tv, alpha_tv
