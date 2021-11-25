#!/usr/bin/env python
""" imgprocess: for image processing
"""

import numpy as np
import skimage
import skimage.feature

#=========================
# basics
#=========================

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


#=========================
# hessian
#=========================

def features_hessian(img, sigma):
    """ stickness and orientation based on 2d Hesssian
    """
    # hessian 2d
    Hrr, Hrc, Hcc = skimage.feature.hessian_matrix(
        img, sigma=sigma, mode="wrap"
    )

    # eigenvalues: l2+, l2-
    # eigenvectors: e2+, e2-
    # mask: select edge-like where l- > |l+|
    mask_tr = (Hrr+Hcc) < 0
    # saliency: |l+ - l-|
    S = mask_tr*np.sqrt((Hrr-Hcc)**2 + 4*Hrc**2)
    # orientation: e- (normal) -> pi/2 rotation (tangent) -> e+
    O = mask_tr*0.5*np.angle(Hrr-Hcc+2j*Hrc)
    return S, O

def features_hessian2(img, sigma):
    """ stickness and orientation based on 2d Hesssian^2
    """
    # hessian 2d
    Hrr, Hrc, Hcc = skimage.feature.hessian_matrix(
        img, sigma=sigma, mode="wrap"
    )
    # hessian*hessian
    H2rr = Hrr*Hrr + Hrc*Hrc
    H2rc = (Hrr+Hcc)*Hrc
    H2cc = Hcc*Hcc + Hrc*Hrc

    # eigenvalues: l2+, l2-
    # eigenvectors: e2+, e2-
    # mask: select edge-like where l- > |l+|
    mask_tr = (Hrr+Hcc) < 0
    # saliency: l2+ - l2- = l+^2 - l-^2
    S2 = mask_tr*np.sqrt((H2rr-H2cc)**2 + 4*H2rc**2)
    # orientation: e2+ (normal) -> pi/2 rotation (tangent) -> e2-
    O2 = mask_tr*0.5*np.angle(-H2rr+H2cc-2j*H2rc)
    return S2, O2
