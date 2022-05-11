
""" B-Spline tools.

References:
    centripetal parameters: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-centripetal.html
    centripetal knots: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-knot-generation.html
"""

import numpy as np
import splipy
from splipy import curve_factory, surface_factory

#=========================
# parametrization
#=========================


class Centripetal:
    """ Centripetal methods for parametrization.
    """
    def __init__(self, degree, exponent=0.5):
        """ Initialization: save parameters.

        Args:
            degree (int): Degree of spline basis.
            exponent (float): Exponent in segment length for parametrization. 0.5 for centripetal, 1 for chordal.
        """
        self.degree = degree
        self.exponent = exponent

    def parametrize1d(self, pts):
        """ Generate parameters for 1d curve.

        Args:
            pts (np.ndarray): Points, with shape=(npts,dim).

        Returns:
            ts (np.ndarray): Parameters for control points, with shape=(npts).
        """
        # parameter selection
        pts = np.asarray(pts)
        # sqrt(distance) for each segment
        l_seg = np.linalg.norm(
            np.diff(pts, axis=0), axis=-1
        )**self.exponent
        # centripedal parameters
        l_accum = np.cumsum(l_seg, axis=0)
        ts = l_accum/l_accum[-1]
        # prepend initial t=0
        ts = np.insert(ts, 0, 0)
        return ts

    def parametrize2d(self, pts_net):
        """ Generate parameters for 2d surface as averages along axes.

        Args:
            pts_net (np.ndarray): Points arranged in a net-shape, with shape=(nu,nv,dim).

        Returns:
            ts_u, ts_v (np.ndarray): Parameters for control points in u,v directions.ts_u.shape=(nu,), ts_v.shape=(nv,).
        """
        nu, nv = pts_net.shape[:2]
        # parameters for u: calc for each v, then avg
        ts_u = np.mean([
            self.parametrize1d(pts_net[:, iv])
            for iv in range(nv)
        ], axis=0)
        # parameters for v: calc for each u, then avg
        ts_v = np.mean([
            self.parametrize1d(pts_net[iu, :])
            for iu in range(nu)
        ], axis=0)
        return ts_u, ts_v

    def generate_knots(self, ts):
        """ Generate knots given parameters.

        Args:
            ts (np.ndarray): Parameters, with shape=(npts,).
        
        Returns:
            knots (np.ndarray): Knots, with shape=(npts+degree+1).
        """
        # knot generation
        npts = len(ts)
        # starting part: degree+1 knots
        knots = np.zeros(npts+self.degree+1)
        # middle part: npts-degree-1 knots
        t_rollsum = np.convolve(
            ts[1:], np.ones(self.degree), mode='valid'
        )[:npts-self.degree-1]
        knots[self.degree+1:npts] = t_rollsum/self.degree
        # ending part: degree+1 knots
        knots[npts:] = 1
        return knots


#=========================
# curve fitting
#=========================

class Curve:
    """ Spline curve.
    
    Used centripetal parametrization method.
    
    Examples:
        curv = Curve(degree=2)
        fit = curv.interpolate(pts)
        pt_i = fit(u_i)
    """

    def __init__(self, degree):
        """ Initialize. Saves degree, construct Centripetal.

        Args:
            degree (int): Spline degree.
        """
        self.degree = degree
        self.param = Centripetal(degree, exponent=0.5)

    def interpolate(self, pts):
        """ Interpolate spline surface.

        Args:
            pts (np.ndarray): Coordinates of points, arranged in shape (nu,dim).

        Returns:
            fit (splipy.curve.Curve): Interpolated curve.
        """
        ts = self.param.parametrize1d(pts)
        knots = self.param.generate_knots(ts)
        basis = splipy.BSplineBasis(order=self.degree+1, knots=knots)
    
        fit = curve_factory.interpolate(
            pts, basis=basis, t=ts
        )
        return fit


#=========================
# surface fitting
#=========================

class Surface:
    """ Spline surface.
    
    Used centripetal parametrization method.
    
    Examples:
        surf = Surface(degree=2)
        fit = surf.interpolate(pts_net)
        pt_i = fit(u_i, v_i)
    """
    def __init__(self, degree):
        """ Initialize. Saves degree, construct Centripetal.

        Args:
            degree (int): Spline degree.
        """
        self.degree = degree
        self.param = Centripetal(degree, exponent=0.5)
        
    def interpolate(self, pts_net):
        """ Interpolate spline surface.

        Args:
            pts_net (np.ndarray): Coordinates of points, arranged in shape (nu,nv,dim).

        Returns:
            fit (splipy.surface.Surface): Interpolated surface.
        """
        ts_u, ts_v = self.param.parametrize2d(pts_net)
        knots_u = self.param.generate_knots(ts_u)
        knots_v = self.param.generate_knots(ts_v)
        basis_u = splipy.BSplineBasis(order=self.degree+1, knots=knots_u)
        basis_v = splipy.BSplineBasis(order=self.degree+1, knots=knots_v)

        fit = surface_factory.interpolate(
            pts_net,
            bases=(basis_u, basis_v),
            u=(ts_u, ts_v)
        )
        return fit
