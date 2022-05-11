""" Utilities for dealing with image-like data.
"""
import numpy as np
import scipy as sp
import pandas as pd
import skimage
import itertools

__all__ = [
    # basics
    "scale_zscore", "scale_minmax", "gaussian",
    # hessians
    "hessian2d", "hessian3d", "symmetric_image",
    # orientations
    "orient_absdiff",
    # morphology
    "connected_components", "component_contour",
    # sparse
    "sparsify3d", "densify3d",
]


#=========================
# basic processing
#=========================

def scale_zscore(I):
    """ Scale an image by z-score.
    
    Return the scaled image if std>0, else return the original image.
    
    Args:
        I (np.ndarray): Image in 2d or 3d.
    
    Returns:
        I_scaled (np.ndarray): The scaled image.
    """
    std = np.std(I)
    if std == 0:
        return I
    else:
        return (I-np.mean(I))/std

def scale_minmax(I, qrange=(0, 1), vrange=(0, 1)):
    """ Scale an image by minmax.

    Return the scaled image if max-min>0, else return the original image.

    Args:
        I (np.ndarray): Image in 2d or 3d.
        qrange (2-tuple): Clip I to this quantile range.
        vrange (2-tuple): The target range of values.

    Returns:
        I_scaled (np.ndarray): The scaled image.
    """
    # calc quantiles
    I_min = np.quantile(I, qrange[0])
    I_max = np.quantile(I, qrange[1])
    I_diff = I_max - I_min

    if I_diff == 0:
        return I
    else:
        I_clipped = np.clip(I, I_min, I_max)
        I_scaled = vrange[0] + (I_clipped-I_min)/I_diff*(vrange[1]-vrange[0])
        return I_scaled

def gaussian(I, sigma):
    """ Gaussian filtering of the image.
    
    A wrapper of skimage.filters.gaussian. If sigma=0, return I.

    Args:
        I (np.ndarray): Image in 2d or 3d.
        sigma (float): Standard deviation for gaussian kernel.
    
    Returns:
        I_gaussian (np.ndarray): The gaussian-filtered image.
    """
    if sigma == 0:
        return I
    else:
        return skimage.filters.gaussian(I, sigma, mode="nearest")


#=========================
# hessians
#=========================

def hessian2d(I, sigma):
    """ Calculate hessian of a 2d image.

    Args:
        I (np.ndarray): 2d image, shape=(ny,nx).
        sigma (float): Sigma for gaussian smoothing (no smoothing if sigma=0).
    
    Returns:
        Hxx, Hxy, Hyy (np.ndarray for each): Elements of the hessian, each has shape=(ny,nx).
    """
    # referenced skimage.feature.hessian_matrix
    # here x,y are made explicit: I[y,x]
    Ig = gaussian(I, sigma)
    Hx = np.gradient(Ig, axis=1)
    Hy = np.gradient(Ig, axis=0)
    Hxx = np.gradient(Hx, axis=1)
    Hxy = np.gradient(Hx, axis=0)
    Hyy = np.gradient(Hy, axis=0)
    return Hxx, Hxy, Hyy

def hessian3d(I, sigma):
    """ Calculate hessian of a 3d image.

    Args:
        I (np.ndarray): 3d image, shape=(nz,ny,nx).
        sigma (float): Sigma for gaussian smoothing (no smoothing if sigma=0).

    Returns:
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz (np.ndarray for each): Elements of the hessian, each has shape=(nz,ny,nx).
    """
    # referenced skimage.feature.hessian_matrix
    # here x,y,z are made explicit: I[z,y,x]
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

def symmetric_image(H_elems):
    """ Make symmetric image from elements
    
    A copy of skimage.feature.corner._symmetric_image.

    Args:
        H_elems (tuple or list): (Hxx,Hxy,Hyy) for 2d, (Hxx,Hxy,Hxz,Hyy,Hyz,Hzz) for 3d.
    
    Returns:
        H_mats (np.ndarray): For 2d, H_mats[y,x]=2d hessian matrix at (y,x). For 3d, H_mats[z,y,x]=3d hessian matrix at (z,y,x).
    """
    shape = H_elems[0].shape
    ndim = H_elems[0].ndim
    dtype = H_elems[0].dtype

    # assign each element of the hessianm matrix
    H_mats = np.zeros(shape + (ndim, ndim), dtype=dtype)
    iter_rc = itertools.combinations_with_replacement(
        range(ndim), 2)
    for idx, (row, col) in enumerate(iter_rc):
        H_mats[..., row, col] = H_elems[idx]
        H_mats[..., col, row] = H_elems[idx]
    return H_mats


#=========================
# orientation tools
#=========================

def orient_absdiff(O1, O2):
    """ Absolute differences between two orientations.

    dO = mod(O2-O1,pi), then wrapped to [0,pi/2) by taking values>pi/2 to pi-values.

    Args:
        O1, O2 (np.ndarray): Two orientations, with values in (-pi/2,pi/2)+n*pi.
    
    Returns:
        dO (np.ndarray): Absolute difference, with the same shape as O1 (or O2).
    """
    dO = np.mod(O2-O1, np.pi)
    dO = np.where(dO<=np.pi/2, dO, np.pi-dO)
    return dO


#=========================
# morphology
#=========================

def connected_components(B, n_keep=None, connectivity=2):
    """ Extract n_keep largest connected components. An iterator.

    Args:
        B (np.ndarray): Binary image, with shape=(ny,nx) or (nz,ny,nx).
        n_keep (int): The number of components to keep.
        connectivity (int): Defines neighboring. E.g. 1 for -|, 2 for -|\/. Range from 1 to B.ndim.

    Yields:
        (size_i, B_i): The size (size_i, int) and binary image (B_i, np.ndarray, shape=B.shape) of each component.
    """
    # label
    L = skimage.measure.label(B, connectivity=connectivity)
    # count
    df = (pd.Series(L[L > 0])
          .value_counts(sort=True, ascending=False)
          .to_frame("size").reset_index()
          )
    # yield
    for item in df.iloc[:n_keep].itertuples():
        B_i = B * (L == item.index)
        yield (item.size, B_i)

def component_contour(B, erode=True):
    """ Get the contour of a connected component.
    
    Contour is the largest one in each plane.

    Args:
        B (np.ndarray): 2d (ny,nx) or 3d (nz,ny,nx) binary image.
        erode (bool): Whether to erode the image first. Erosion avoids broken contours due to the boundary.
    
    Returns:
        contour (np.ndarray): Shape=(npts,ndim). [[y0,x0],...] for 2d, [[z0,y0,x0],...] for 3d.
    """
    def get_largest(contours):
        # find the largest of an array of contours
        sizes = [len(c) for c in contours]
        imax = np.argmax(sizes)
        return contours[imax]

    # 2d case
    if B.ndim == 2:
        if erode:
            B = skimage.morphology.binary_erosion(B)
        contour = get_largest(skimage.measure.find_contours(B))
    
    # 3d case
    elif B.ndim == 3:
        contour = []
        for i, B_i in enumerate(B):
            if erode:
                B_i = skimage.morphology.binary_erosion(B_i)
            yx_i = get_largest(skimage.measure.find_contours(B_i))
            zyx_i = np.concatenate(
                [i*np.ones((len(yx_i), 1)), yx_i], axis=1)
            contour.append(zyx_i)
        contour = np.concatenate(contour, axis=0)
    return contour

#=========================
# sparse tools
#=========================

def sparsify3d(I):
    """ Sparsify 3d image to an array of coo_matrix.

    Args:
        I (np.ndarray): 3d image.
    
    Returns:
        I_sparse (np.ndarray of sp.sparse.coo_matrix): Sparsified stack of images.
    """
    I_sparse = np.array(
        [sp.sparse.coo_matrix(I[i]) for i in range(len(I))]
    )
    return I_sparse

def densify3d(I_sparse):
    """ Recover 3d image from its sparsified format.

    Args:
        I_sparse (np.ndarray of sp.sparse.coo_matrix): Sparsified stack of images, which can be produced from imgutils.sparsify3d.

    Returns:
        I (np.ndarray): 3d image.
    """
    I = np.array(
        [I_sparse[i].todense()
        for i in range(len(I_sparse))
    ])
    return I
