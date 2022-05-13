""" Steerable tensor voting (TV).

SteerTV:
    notation c.f. Franken2006
    wm = exp(-r**2/(2*sigma**2))*((x+iy)/r)**m
    cm = S*exp(-i*m*O)

    stick field: S=|U|, O=arg(U)/2
    U = w0*c2c + 4*w2*c0 + 6*w4*c2 + 4*w6*c4 + w8*c6
    *: convolution, A*B = ifft(fft(A)*fft(B))

    ball field:
    B = Re[6*w0*c0 + 8*w2*c2 + 2*w4*c4]

References:
    TomoSegMemTV: Martinez-Sanchez, A.; Garcia, I.; Asano, S.; Lucic, V.; Fernandez, J.-J. Robust Membrane Detection Based on Tensor Voting for Electron Tomography. Journal of Structural Biology 2014, 186 (1), 49–61. https://doi.org/10.1016/j.jsb.2014.02.015.
    SteerTV: Franken, E.; van Almsick, M.; Rongen, P.; Florack, L.; ter Haar Romeny, B. An Efficient Method for Tensor Voting Using Steerable Filters. In Computer Vision – ECCV 2006; Leonardis, A., Bischof, H., Pinz, A., Eds.; Lecture Notes in Computer Science; Springer Berlin Heidelberg: Berlin, Heidelberg, 2006; Vol. 3954, pp 228–240. https://doi.org/10.1007/11744085_18.
"""

import numpy as np
from numpy import fft
import multiprocessing.dummy
from etsynseg import imgutils

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

def prep_wmfft(ny, nx, sigma, m, r2_exclude, value_r0, n_terms):
    """ Prepare fft of wm. Radial function f(r)=r^m e^(-r^2/2sigma^2).

    Args:
        ny, nx (int): The number of rows(y) and columns(x).
        sigma, m (float): Parameters of the radial function f(r).
        r2_exclude (float): Regions around the origin (r^2<r2_exclude) are set to zero.
        value_r0 (float): Value at the origin, to avoid nan due to singularity.
        n_terms (int): Take the first n terms in wmfft. 5 for stick field, 3 for ball field.
    
    Returns:
        wmfft (np.ndarray): The fft of [w0,w2,w4,w6,w8][:n_terms], with shape of wi=(ny,nx).
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
    term_exp[r2<r2_exclude] = 0

    # fraction term
    r2[0, 0] = 1  # avoid division by zero
    term_frac2 = (x+1j*y)**2/r2
    # set origin to avoid nan
    term_frac2[0, 0] = value_r0

    # calculate wm accumulatively
    wmfft = np.zeros((n_terms, ny, nx), dtype=np.complex_)
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
    """ Prepare fft of wm for stick field.
    
    f(r)=e^(-r^2/2sigma^2).
    value_r0=1.

    r2_exclude is set to 1, due to some steertv issues at r=0.

    Args:
        ny, nx (int): The number of rows(y) and columns(x).
        sigma (float): Controls the decay in radial function.

    Returns:
        wmfft (np.ndarray): fft of [w0,w2,w4,w6,w8], shape=(5,ny,nx).
    """
    wmfft = prep_wmfft(ny, nx, sigma,
        m=0, r2_exclude=1, value_r0=1, n_terms=5
    )
    return wmfft

def stick2d_wmfft(S, O, wmfft):
    """ Stick field TV for 2d slice. Given wmfft.

    Args:
        S, O (np.ndarray): Input saliency and orientation, with shape=(ny,nx).
        wmfft (np.ndarray): fft of [w0,w2,w4,w6,w8].

    Returns:
        Stv, Otv (np.ndarray): Saliency and orientation after TV, with shape=(ny,nx).
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
    Stv = np.abs(U)
    Otv = 0.5*np.angle(U)
    return Stv, Otv

def stick2d(S, O, sigma):
    """ Stick field TV for 2d slice.
    
    f(r)=e^(-r^2/2sigma^2).
    value_r0=1.

    Args:
        S, O (np.ndarray): Input saliency and orientation, with shape=(ny,nx).
        sigma (float): Controls the decay in radial function.

    Returns:
        Stv, Otv (np.ndarray): Saliency and orientation after TV, with shape=(ny,nx).
    """
    ny, nx = S.shape
    wmfft = prep_wmfft_stick(ny, nx, sigma)
    Stv, Otv = stick2d_wmfft(S, O, wmfft)
    return Stv, Otv

def stick3d(S, O, sigma):
    """ Stick field TV for stack of 2d images.
    
    f(r)=e^(-r^2/2sigma^2).
    value_r0=1.
    Used multithreading.

    Args:
        S, O (np.ndarray): Input saliency and orientation, with shape=(nz,ny,nx).
        sigma (float): Controls the decay in radial function.

    Returns:
        Stv, Otv (np.ndarray): Saliency and orientation after TV, with shape=(nz,ny,nx).
    """
    # prep
    nz, ny, nx = S.shape
    wmfft = prep_wmfft_stick(ny=ny, nx=nx, sigma=sigma)

    # calc S, O for each slice
    Stv = np.zeros(S.shape, dtype=np.float_)
    Otv = np.zeros(O.shape, dtype=np.float_)

    def calc_one(i):
        Stv[i], Otv[i] = stick2d_wmfft(S[i], O[i], wmfft)

    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(nz))
    pool.close()
    return Stv, Otv


#=========================
# ball field
#=========================

def prep_wmfft_ball(ny, nx, sigma):
    """ Prepare fft of wm for ball field.
    
    f(r)=e^(-r^2/2sigma^2).
    value_r0=0.
    r2_exclude=3, to avoid normalsup of membrane pixels.

    Args:
        ny, nx (int): The number of rows(y) and columns(x).
        sigma (float): Controls the decay in radial function.

    Returns:
        wmfft (np.ndarray): fft of [w0,w2,w4], shape=(3,ny,nx).
    """
    wmfft = prep_wmfft(ny, nx, sigma,
        m=0, r2_exclude=3, value_r0=0, n_terms=3
    )
    return wmfft

def ball2d_wmfft(S, O, wmfft):
    """ Ball field TV for 2d slice. Given wmfft.

    Args:
        S, O (np.ndarray): Input saliency and orientation, with shape=(ny,nx).
        wmfft (np.ndarray): fft of [w0,w2,w4].

    Returns:
        Stv (np.ndarray): Saliency after TV, with shape=(ny,nx).
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
    Stv = np.sum(series, axis=0)
    return Stv

def ball2d(S, O, sigma):
    """ Ball field TV for 2d slice.
    
    f(r)=e^(-r^2/2sigma^2).
    value_r0=0.
    r2_exclude=3, to avoid normalsup of membrane pixels.

    Args:
        S, O (np.ndarray): Input saliency and orientation, with shape=(ny,nx).
        sigma (float): Controls the decay in radial function.

    Returns:
        Stv (np.ndarray): Saliency after TV, with shape=(ny,nx).
    """
    ny, nx = S.shape
    wmfft = prep_wmfft_ball(ny, nx, sigma)
    Stv = ball2d_wmfft(S, O, wmfft)
    return Stv

def ball3d(S, O, sigma):
    """ Ball field TV for stack of 2d images.
    
    f(r)=e^(-r^2/2sigma^2).
    value_r0=0.
    r2_exclude=3, to avoid normalsup of membrane pixels.
    Used multithreading.

    Args:
        S, O (np.ndarray): Input saliency and orientation, with shape=(nz,ny,nx).
        sigma (float): Controls the decay in radial function.

    Returns:
        Stv (np.ndarray): Saliency after TV, with shape=(nz,ny,nx).
    """
    # prep
    nz, ny, nx = S.shape
    wmfft = prep_wmfft_ball(ny=ny, nx=nx, sigma=sigma)
    
    # calc S, O for each slice
    Stv = np.zeros(S.shape, dtype=np.float_)

    def calc_one(i):
        Stv[i] = ball2d_wmfft(S[i], O[i], wmfft)

    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(nz))
    pool.close()
    return Stv


#=========================
# suppress by orientation
#=========================

def suppress_by_orient(B, O, sigma, dO_thresh=np.pi/4):
    """ Apply a strong TV field, then suppress pixels where change in O is large.

    Args:
        B, O (np.ndarray): Input binary image and orientation, with shape=(nz,ny,nx).
        dO_thresh (float, in rad): Suppress pixels whose change in O after TV is >= this value.

    Returns:
        Bsupp (np.ndarray): Binary mask of image with 1's at non-suppressed pixels and 0's otherwise, with shape=(nz,ny,nx).
        Stv (np.ndarray): Saliency after TV, with shape=(nz,ny,nx).
    """
    # apply strong tv field
    Stv, Otv = stick3d(B, O, sigma)
    # calculate change in O
    dO = imgutils.orients_absdiff(Otv, O)
    # mask of pixels with small dO
    Bsupp = B*(dO<dO_thresh)
    return Bsupp, Stv
