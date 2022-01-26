
import numpy as np
from scipy import optimize
import splipy.surface_factory

#=========================
# universal method
#=========================
class Surface:
    """ surface using universal parametrization method
    usage:
        surf = Surface(uv_size=(nu, nv), degree=2)
        fit = surf.interpolate(pts_net, degree=2)
    """
    def __init__(self, uv_size, degree):
        """ init basis and ts
        """
        # basic info
        self.uv_size = uv_size
        self.degree = degree
        # parametrization, basis
        self.knots_u, self.basis_u, self.ts_u = Surface.parametrize(
            n_ctrl=uv_size[0], degree=degree
        )
        self.knots_v, self.basis_v, self.ts_v = Surface.parametrize(
            n_ctrl=uv_size[1], degree=degree
        )
        # assemble
        self.basis_uv = [self.basis_u, self.basis_v]
        self.ts_uv = [self.ts_u, self.ts_v]

    @staticmethod
    def parametrize(n_ctrl, degree):
        """ universal parametrization
        :param n_ctrl: number of control points
        :param degree: degree
        :return: knots, basis, ts
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

    def interpolate(self, pts_net):
        """ interpolate surface
        :param pts_net: shape=(nu,nv,dim)
        :param degree: degree of bspline in u,v directions
        :return: fit
            fit: splipy surface
        """
        fit = splipy.surface_factory.interpolate(
            pts_net, bases=self.basis_uv, u=self.ts_uv
        )
        return fit


#=========================
# centripetal method
#=========================

# class SurfaceCentripetal:
#     """ surface using centripetal parametrization method
#     usage:
#         fit = SurfaceCentripetal.interpolate(pts_net, degree=2)
#     """
#     @staticmethod
#     def parametrize_oneaxis(pts_net, degree, exponent):
#         """ generate parameters (ts) and knots
#         results are 1d arrays along axis=0, averaged over axis=1
#         :param pts_net: shape=(nu,nv,dim)
#         :param degree: degree of bspline
#         :param exponent: chordal - 1; centripetal - 0.5
#         :return: ts, knots
#         """
#         pts_net = np.asarray(pts_net)

#         # parameter selection
#         # sqrt(distance) for each segment
#         l_seg = np.linalg.norm(np.diff(pts_net, axis=0), axis=-1)**exponent
#         # centripedal parameters
#         l_accum = np.cumsum(l_seg, axis=0)
#         ts = np.mean(l_accum/l_accum[-1, :], axis=1)
#         # prepend initial t=0
#         ts = np.insert(ts, 0, 0)

#         # knot generation
#         n_data = len(pts_net)
#         # starting part: degree+1 knots
#         knots = np.zeros(n_data+degree+1)
#         # middle part: n_data-degree-1 knots
#         t_rollsum = np.convolve(ts[1:], np.ones(
#             degree), mode='valid')[:n_data-degree-1]
#         knots[degree+1:n_data] = t_rollsum/degree
#         # ending part: degree+1 knots
#         knots[n_data:] = 1
#         return ts, knots

#     @staticmethod
#     def parametrize(pts_net, degree, exponent):
#         """ generate parameters and knots
#         :param pts_net: shape=(nu,nv,dim)
#         :param degree: degree of bspline in u,v directions
#         :param exponent: chordal - 1; centripetal - 0.5
#         :return: ts_uv, knots_uv
#             ts_uv: (ts_u, ts_v)
#             knots_uv: (knots_u, knots_v)
#         """
#         # u-direction
#         ts_u, knots_u = SurfaceCentripetal.parametrize_oneaxis(pts_net, degree, exponent)
#         # v-direction
#         pts_swap = np.transpose(pts_net, axes=(1, 0, 2))
#         ts_v, knots_v = SurfaceCentripetal.parametrize_oneaxis(pts_swap, degree, exponent)
#         # return
#         ts_uv = (ts_u, ts_v)
#         knots_uv = (knots_u, knots_v)
#         return ts_uv, knots_uv

#     @staticmethod
#     def interpolate(pts_net, degree, exponent=0.5):
#         """ interpolate surface
#         :param pts_net: shape=(nu,nv,dim)
#         :param degree: degree of bspline in u,v directions
#         :param exponent: chord length method - 1; centripetal - 0.5
#         :return: fit
#             fit: splipy surface
#         """
#         # centripetal method
#         ts_uv, knots_uv = SurfaceCentripetal.parametrize(pts_net, degree, exponent)
#         bases = [
#             splipy.BSplineBasis(order=degree+1, knots=knots)
#             for knots in knots_uv
#         ]
#         # fit
#         fit = splipy.surface_factory.interpolate(
#             pts_net, bases=bases, u=ts_uv
#         )
#         return fit
