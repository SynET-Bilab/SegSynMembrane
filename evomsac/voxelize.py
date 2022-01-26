
import functools
import numpy as np

class Voxelize:
    """ Pixel-precision voxelization of a parameteric surface (from NURBS fitting)
    Usage:
        vox = Voxelize(surface)
        vox.run([0, 1], [0, 1])
        vox.pts
    Attributes:
        pts: coordinates of surface points, [[z1,y1,x1],[z2,y2,x2],...]
        uv: parameters of surface points, [[u1,v1],[u2,v2],...]
    """
    def __init__(self, surface):
        """ init
        """
        self.surface = surface
        self.pts = []
        self.uv = []
        self.evaluate.cache_clear()

    @functools.lru_cache
    def evaluate(self, u, v):
        """ cached evaluation of surface at location (u, v)
        :param u, v: parameteric positions
        :return: coord
            coord: [z,y,x] at (u,v)
        """
        return self.surface(u, v)

    def run(self, bound_u, bound_v):
        """ recursive rasterization surface in a uv-square
        :param bound_u, bound_v: bounds in u,v directions, e.g. [u_min, u_max]
        :return: None
        :action:
            if corners are not connected: run for finer regions
            if corners are connected: append midpoint to self.pts
        """
        # evaluate corners, calculate differences
        fuv = [[self.evaluate(u, v) for v in bound_v] for u in bound_u]
        diff_v = np.abs(np.diff(fuv, axis=1)).max() > 1
        diff_u = np.abs(np.diff(fuv, axis=0)).max() > 1

        # calculate midpoints
        mid_u = np.mean(bound_u)
        mid_v = np.mean(bound_v)

        # if corners in both u,v directions have gaps
        # divide into 4 sub-regions and run for each
        if diff_v and diff_u:
            self.run([bound_u[0], mid_u], [bound_v[0], mid_v])
            self.run([bound_u[0], mid_u], [mid_v, bound_v[1]])
            self.run([mid_u, bound_u[1]], [bound_v[0], mid_v])
            self.run([mid_u, bound_u[1]], [mid_v, bound_v[1]])
        # if corners in v directions have gaps
        # divide into 2 sub-regions and run for each
        elif diff_v:
            self.run(bound_u, [bound_v[0], mid_v])
            self.run(bound_u, [mid_v, bound_v[1]])
        # if corners in u directions have gaps
        # divide into 2 sub-regions and run for each
        elif diff_u:
            self.run([bound_u[0], mid_u], bound_v)
            self.run([mid_u, bound_u[1]], bound_v)
        # if no corners have gaps
        # append midpoint and its value to results
        else:
            f_mid = np.mean(fuv, axis=(0, 1))
            self.pts.append(f_mid.tolist())
            self.uv.append([mid_u, mid_v])
            return

