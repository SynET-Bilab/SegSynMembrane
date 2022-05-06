
import numpy as np
from scipy import optimize
import splipy
from splipy import curve_factory, surface_factory

#=========================
# parametrization: universal method
#=========================

def parametrize(n_ctrl, degree):
    """Universal parametrization.

    Args:
        n_ctrl (int): The number of control points.
        degree (int): The degree of spline.

    Returns:
        knots (np.ndarray): Knots, 1d array.
        basis (splipy.basis.BSplineBasis): Basis to interpolate on.
        ts (np.ndarray): Parametric interpolation points, 1d array.
    """
    # knots: clamped, equidistant
    n_knots = n_ctrl + degree + 1
    knots = np.zeros(n_knots)
    knots[degree+1:n_ctrl] = np.arange(1, n_ctrl-degree)/(n_ctrl-degree)
    knots[n_ctrl:] = 1

    # parameters: max at bases
    basis = splipy.BSplineBasis(order=degree+1, knots=knots)
    ts = np.zeros(n_ctrl)
    for i in range(n_ctrl):
        def func(x): 
            return -basis.evaluate(x[0])[0][i]
        ts[i] = optimize.brute(
            func, ranges=[(0, 1)], Ns=n_ctrl*2,
            finish=optimize.fmin
        )[0]

    return knots, basis, ts

#=========================
# curve fitting
#=========================
class Curve:
    """ Spline curve using universal parametrization method.
    
    Examples:
        curv = Curve(nu, degree=2)
        fit = curv.interpolate(pts)
        pt_i = fit(u_i)
    """

    def __init__(self, nu, degree):
        """ Initialize basis and ts.

        Args:
            nu (int): The number of control points.
            degree (int): Spline degree.
        """
        # basic info
        self.nu = nu
        self.degree = degree
        # parametrization, basis
        self.knots_u, self.basis_u, self.ts_u = parametrize(
            n_ctrl=nu, degree=degree
        )

    def interpolate(self, pts):
        """ Interpolate spline surface.

        Args:
            pts (np.ndarray): Coordinates of points, arranged in shape (nu,dim).

        Returns:
            fit (splipy.curve.Curve): Interpolated curve.
        """
        fit = curve_factory.interpolate(
            pts, basis=self.basis_u, t=self.ts_u
        )
        return fit

#=========================
# surface fitting
#=========================
class Surface:
    """ Spline surface using universal parametrization method.
    
    Examples:
        surf = Surface(nu, nv, degree=2)
        fit = surf.interpolate(pts_net)
        pt_i = fit(u_i, v_i)
    """
    def __init__(self, nu, nv, degree):
        """ Initialize basis and ts.

        Args:
            nu (int): The number of control points in u-direction.
            nv (int): The number of control points in v-direction.
            degree (int): Spline degree.
        """
        # basic info
        self.nu = nu
        self.nv = nv
        self.degree = degree
        # parametrization, basis
        self.knots_u, self.basis_u, self.ts_u = parametrize(
            n_ctrl=nu, degree=degree
        )
        self.knots_v, self.basis_v, self.ts_v = parametrize(
            n_ctrl=nv, degree=degree
        )
        # assemble
        self.basis_uv = [self.basis_u, self.basis_v]
        self.ts_uv = [self.ts_u, self.ts_v]

    def interpolate(self, pts_net):
        """ Interpolate spline surface.

        Args:
            pts_net (np.ndarray): Coordinates of points, arranged in shape (nu,nv,dim).

        Returns:
            fit (splipy.surface.Surface): Interpolated surface.
        """
        fit = surface_factory.interpolate(
            pts_net, bases=self.basis_uv, u=self.ts_uv
        )
        return fit
