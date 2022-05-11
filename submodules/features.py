""" Features in the image.
"""

import numpy as np
import multiprocessing.dummy
from etsynseg import imgutils

__all__ = [
    "ridgelike2d", "ridgelike3d",
]

def ridgelike2d(I, sigma):
    """ Ridge-like features of 2d image.
    
    Stickness and orientation based on 2d hesssian^2

    Args:
        I (np.ndarray): 2d image, shape=(ny,nx).
        sigma (float): Sigma for gaussian smoothing.

    Returns:
        S (np.ndarray): Stickness saliency (>0). The same shape as I.
        O (np.ndarray): Orientation of the sticks. Ranged within (-pi,pi]. The same shape as I.
    """
    # hessian 2d
    Hxx, Hxy, Hyy = imgutils.hessian2d(I, sigma)
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

def ridgelike3d(I, sigma):
    """ Ridge-like features of stack of 2d images.
    
    Stickness and orientation based on 2d hesssian^2 for each slice.

    Args:
        I (np.ndarray): 3d image, shape=(nz,ny,nx).
        sigma (float): Sigma for gaussian smoothing.

    Returns:
        S (np.ndarray): Stickness saliency (>0). The same shape as I.
        O (np.ndarray): Orientation of the sticks. Ranged within (-pi,pi]. The same shape as I.
    """
    # gaussian on 3d image
    Ig = imgutils.gaussian(I.astype(np.float_), sigma)

    # create arrays
    S = np.zeros(Ig.shape, dtype=np.float_)
    O = np.zeros(Ig.shape, dtype=np.float_)
    nz = S.shape[0]

    # parallel computing
    def calc_one(i):
        S[i], O[i] = ridgelike2d(Ig[i], 0)

    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(nz))
    pool.close()
    return S, O
