#!/usr/bin/env python
""" dtvoting: for dense tensor voting

SteerTV:
    km[m] = exp(-2j*(m-1)*O)
    U = sum([ifft2(fft2(km[m]*S)*vmfft[m] for m in range(5)])
    S_tv = abs(U)
    O_tv = 0.5*angle(U)

Methods:
    precalc_vmfft(nrow, ncol, sigma) -> vmfft
    tv2d(S, O, vmfft) -> S_tv[y,x], O_tv[y,x]
    tv3d(S, O, sigma, method="numba") -> S_tv[z,y,x], O_tv[z,y,x]
"""

import numpy as np
import numpy.fft as fft
import numba
# import functools
# import pathlib
# import tempfile
# import dask
# import dask.distributed


#=========================
# tv2d
#=========================

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

def tv2d(S, O, vmfft):
    """ tv for 2d slice
    param S, O: shape=(ny,nx)
    param vmfft: array of 5 elements, each with shape=(ny,nx)
    return: S_tv, O_tv
    """
    # orientation term
    km_234 = [np.exp(-2j*(m-1)*O) for m in [2, 3, 4]]
    km_01 = [np.conj(km_234[0]), 1]
    km = km_01 + km_234
    
    # sum the series
    series = [
        fft.ifft2(fft.fft2(km[m]*S)*vmfft[m])
        for m in range(5)
    ]
    U = np.sum(series, axis=0)

    # calculate saliency and orientation
    S_tv = np.abs(U)
    O_tv = 0.5*np.angle(U)
    return S_tv, O_tv


#=========================
# tv3d
#=========================

def tv3d_python(S, O, sigma):
    """ tv for 3d volume
    return: S_tv, O_tv
    """
    # precalc vmfft
    nz, ny, nx = S.shape
    vmfft = precalc_vmfft(nrow=ny, ncol=nx, sigma=sigma)

    # calc S, O for each slice
    S_tv = np.zeros_like(S)
    O_tv = np.zeros_like(O)
    for i in range(nz):
        S_tv[i], O_tv[i] = tv2d(S[i], O[i], vmfft)
    return S_tv, O_tv

@numba.njit(parallel=True)
def tv3d_numba(S, O, sigma):
    """ tv for 3d volume, using numba parallel
    return: S_tv, O_tv
    """
    nz, ny, nx = S.shape
    with numba.objmode(vmfft='complex128[:,:,:]'):
        vmfft = precalc_vmfft(nrow=ny, ncol=nx, sigma=sigma)
    # calc S, O for each slice
    S_tv = np.zeros_like(S)
    O_tv = np.zeros_like(O)
    for i in numba.prange(nz):
        with numba.objmode(S_tv_i='float64[:,:]', O_tv_i='float64[:,:]'):
            S_tv_i, O_tv_i = tv2d(S[i], O[i], vmfft)
        S_tv[i] = S_tv_i
        O_tv[i] = O_tv_i
    return S_tv, O_tv

def tv3d(S, O, sigma, method="numba"):
    """ tv for 3d volume
    param S, O: shape=(nz,ny,nx)
    param sigma: scale in pixel
    param method: python or numba
    return: S_tv, O_tv
    """
    if method == "numba":
        return tv3d_numba(S, O, sigma)
    elif method == "python":
        return tv3d_python(S, O, sigma)
    else:
        raise ValueError("method: python or numba")

# def _tv3d_dask_slice(input_paths, i):
#     """ tv for 3d volume, auxiliary function for a 2d slice
#     param input_paths: dict(S,O,vmfft) to path of npys
#     param i: index of slice
#     return: S_tv_i, O_tv_i
#     """
#     # load slice using memmap mode
#     S_i = np.load(input_paths["S"], mmap_mode="r")[i]
#     O_i = np.load(input_paths["O"], mmap_mode="r")[i]
#     vmfft = np.load(input_paths["vmfft"], mmap_mode="r")
#     # tv2d
#     S_tv_i, O_tv_i = tv2d(S_i, O_i, vmfft)
#     return S_tv_i, O_tv_i

# def tv3d_dask(S, O, sigma):
#     """ tv for 3d volume, multiprocessing using dask
#     return: S_tv, O_tv
#     """
#     # precalc vmfft
#     nz, ny, nx = S.shape
#     vmfft = precalc_vmfft(nrow=ny, ncol=nx, sigma=sigma)

#     # save as tempfile npy
#     input_paths = {
#         arr: tempfile.NamedTemporaryFile(
#             delete=False, suffix=".npy").name
#         for arr in ["S", "O", "vmfft"]
#     }
#     np.save(input_paths["S"], S)
#     np.save(input_paths["O"], O)
#     np.save(input_paths["vmfft"], vmfft)

#     # distributed computing using dask
#     # add input_paths to tv2d
#     tv2d_partial = functools.partial(_tv3d_dask_slice, input_paths)
#     # dask: client - map - gather
#     client = dask.distributed.Client()
#     futures = client.map(tv2d_partial, range(nz))
#     results = client.gather(futures)

#     # clean-up
#     # close client
#     client.close()
#     # delete tempfiles
#     for arr in ["S", "O", "vmfft"]:
#         pathlib.Path(input_paths[arr]).unlink(missing_ok=True)

#     # assemble results
#     S_tv = np.zeros_like(S)
#     O_tv = np.zeros_like(O)
#     for i in range(nz):
#         S_tv[i] = results[i][0]
#         O_tv[i] = results[i][1]

#     # return
#     return S_tv, O_tv
