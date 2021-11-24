#!/usr/bin/env python
""" imgprocess: for image processing
"""

import numpy as np
import skimage
import skimage.feature

def zscore(img):
    """ return zscore of img """
    I = np.array(img)
    z = (I-I.mean())/I.std()
    return z

def invert(img):
    """ switch foreground between white and dark
    zscore then invert
    """
    I = np.array(img)
    std = I.std()
    if std > 0:
        return -(I-I.mean())/std
    else:
        return np.zeros_like(I)

def features_hess2d(img, sigma):
    """ stickness and orientation based on 2d Hesssian
    """
    # hessian 2d
    Hrr, Hrc, Hcc = skimage.feature.hessian_matrix(
        img, sigma=sigma, mode="wrap"
    )

    # mask 
    # saliency: |eigval1-eigval2| if trace<0
    mask_tr = (Hrr+Hcc)<0
    S = mask_tr*np.sqrt((Hrr-Hcc)**2 + 4*Hrc**2)

    # 
    O = 0.5*np.angle(Hrr-Hcc+2j*Hrc)

