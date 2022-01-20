#!/usr/bin/env python

import numpy as np
import splipy.surface_factory


def centripetal_oneaxis(pts_net, degree, exponent):
    """ generate parameters (ts) and knots
    results are 1d arrays along axis=0, averaged over axis=1
    :param pts_net: shape=(nu,nv,dim)
    :param degree: degree of bspline
    :param exponent: chord length method - 1; centripetal - 0.5
    :return: ts, knots
    """
    pts_net = np.asarray(pts_net)

    # parameter selection
    # sqrt(distance) for each segment
    l_seg = np.linalg.norm(np.diff(pts_net, axis=0), axis=-1)**exponent
    # centripedal parameters
    l_accum = np.cumsum(l_seg, axis=0)
    ts = np.mean(l_accum/l_accum[-1, :], axis=1)
    # prepend initial t=0
    ts = np.insert(ts, 0, 0)

    # knot generation
    n_data = len(pts_net)
    # starting part: degree+1 knots
    knots = np.zeros(n_data+degree+1)
    # middle part: n_data-degree-1 knots
    t_rollsum = np.convolve(ts[1:], np.ones(
        degree), mode='valid')[:n_data-degree-1]
    knots[degree+1:n_data] = t_rollsum/degree
    # ending part: degree+1 knots
    knots[n_data:] = 1
    return ts, knots

def centripetal_surface(pts_net, degree, exponent):
    """ generate parameters and knots
    :param pts_net: shape=(nu,nv,dim)
    :param degree: degree of bspline in u,v directions
    :param exponent: chord length method - 1; centripetal - 0.5
    :return: ts_uv, knots_uv
        ts_uv: (ts_u, ts_v)
        knots_uv: (knots_u, knots_v)
    """
    # u-direction
    ts_u, knots_u = centripetal_oneaxis(pts_net, degree, exponent)
    # v-direction
    pts_swap = np.transpose(pts_net, axes=(1, 0, 2))
    ts_v, knots_v = centripetal_oneaxis(pts_swap, degree, exponent)
    # return
    ts_uv = (ts_u, ts_v)
    knots_uv = (knots_u, knots_v)
    return ts_uv, knots_uv

def interpolate_surface(pts_net, degree, exponent=0.5):
    """ interpolate surface
    :param pts_net: shape=(nu,nv,dim)
    :param degree: degree of bspline in u,v directions
    :param exponent: chord length method - 1; centripetal - 0.5
    :return: fit
        fit: splipy surface
    """
    # centripetal method
    ts_uv, knots_uv = centripetal_surface(pts_net, degree, exponent)
    bases = [
        splipy.BSplineBasis(order=degree+1, knots=knots)
        for knots in knots_uv
    ]
    # fit
    fit = splipy.surface_factory.interpolate(
        pts_net, bases=bases, u=ts_uv
    )
    return fit
