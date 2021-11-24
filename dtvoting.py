#!/usr/bin/env python
""" dtvoting

Tensor voting:
    km[m] = exp(-2j*(m-1)*alpha)
    Um2 = sum([ifft2(fft2(km[m]*S)*vmfft[m] for m in range(5)])
    S_tv = abs(Um2)
    alpha_tv = 0.5*angle(Um2)

Methods:
    precalc_vmfft(nrow, ncol, sigma) -> vmfft
    tv2d(S, alpha, vmfft) -> S_tv[y,x], alpha_tv[y,x]
    tv3d(S, alpha, sigma) -> S_tv[z,y,x], alpha_tv[z,y,x]
    tv3d_dask(S, alpha, sigma) -> S_tv[z,y,x], alpha_tv[z,y,x]
"""

import functools
import pathlib
import tempfile
import numpy as np
import numpy.fft as fft
import dask
import dask.distributed

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

def tv3d(S, alpha, sigma):
    """ tv for 3d volume
    return: S_tv, alpha_tv
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

def _tv3d_dask_slice(input_paths, i):
    """ tv for 3d volume, auxiliary function for a 2d slice
    param input_paths: dict(S,alpha,vmfft) to path of npys
    param i: index of slice
    return: S_tv_i, alpha_tv_i
    """
    # load slice using memmap mode
    S_i = np.load(input_paths["S"], mmap_mode="r")[i]
    alpha_i = np.load(input_paths["alpha"], mmap_mode="r")[i]
    vmfft = np.load(input_paths["vmfft"], mmap_mode="r")
    # tv2d
    S_tv_i, alpha_tv_i = tv2d(S_i, alpha_i, vmfft)
    return S_tv_i, alpha_tv_i

def tv3d_dask(S, alpha, sigma):
    """ tv for 3d volume, multiprocessing using dask
    return: S_tv, alpha_tv
    """
    # precalc vmfft
    nz, ny, nx = S.shape
    vmfft = precalc_vmfft(nrow=ny, ncol=nx, sigma=sigma)

    # save as tempfile npy
    input_paths = {
        arr: tempfile.NamedTemporaryFile(
            delete=False, suffix=".npy").name
        for arr in ["S", "alpha", "vmfft"]
    }
    np.save(input_paths["S"], S)
    np.save(input_paths["alpha"], alpha)
    np.save(input_paths["vmfft"], vmfft)

    # distributed computing using dask
    # add input_paths to tv2d
    tv2d_partial = functools.partial(_tv3d_dask_slice, input_paths)
    # dask: client - map - gather
    client = dask.distributed.Client()
    futures = client.map(tv2d_partial, range(nz))
    results = client.gather(futures)

    # clean-up
    # close client
    client.close()
    # delete tempfiles
    for arr in ["S", "alpha", "vmfft"]:
        pathlib.Path(input_paths[arr]).unlink(missing_ok=True)

    # assemble results
    S_tv = np.zeros_like(S)
    alpha_tv = np.zeros_like(alpha)
    for i in range(nz):
        S_tv[i] = results[i][0]
        alpha_tv[i] = results[i][1]

    # return
    return S_tv, alpha_tv
