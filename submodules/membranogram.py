""" membranogram
"""
import multiprocessing.dummy
import numpy as np
import scipy as sp
from sklearn import manifold
from etsynseg import pcdutil

__all__ = [
    "interpolate_dist", "interpolate_avg",
    "Embed", "Project"
]

#=========================
# membranogram
#=========================

def interpolate_dist(zyx, nzyx, dist, I):
    """ Generate membranogram.
    
    Interpolate the image at locations of zyx+nzyx*dist.

    Args:
        zyx (np.ndarray): Points with shape=(npts,3), arranged in [[z0,y0,x0],...].
        nzyx (np.ndarray): Normal vectors corresponding to the points. Shape=(npts,3). Arranged in [[nz0,ny0,nx0],...].
        dist (float): Distance from points, which can be negative.
        I (np.ndarray): Image with shape=(nz,ny,nx).

    Returns:
        zyx_d (np.ndarray): Points at dist from membrane.
        v_d (np.ndarray): Values for points in zyx_d, shape=(npts).
    """
    zyx_d = zyx+nzyx*dist
    v_d = sp.ndimage.map_coordinates(
        I, zyx_d.T, order=3, mode="nearest"
    )
    return zyx_d, v_d

def interpolate_distarr(zyx, nzyx, dist_arr, I):
    """ Generate membranogram at an array of distances.

    Args:
        zyx (np.ndarray): Points with shape=(npts,3), arranged in [[z0,y0,x0],...].
        nzyx (np.ndarray): Normal vectors corresponding to the points. Shape=(npts,3). Arranged in [[nz0,ny0,nx0],...].
        dist_arr (list of float): A list of distance from points, whose interpolated values will be averaged.
        I (np.ndarray): Image with shape=(nz,ny,nx).

    Returns:
        v_arr (np.ndarray): Interpolated values at the distances, shape=(ndist,npts).
            v_arr[i]: values at zyx+dist_arr[i]*nzyx
    """
    # init interpolator
    I_interp = sp.interpolate.RegularGridInterpolator(
        [np.arange(I.shape[i]) for i in range(3)], I,
        method="linear"
    )
    # init empty arrays
    v_arr = np.zeros((len(dist_arr), len(zyx)), dtype=float)

    # interpolate at dists
    def calc_one(i):
        zyx_d = zyx + dist_arr[i]*nzyx
        v_arr[i] = I_interp(zyx_d)

    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(len(dist_arr)))
    pool.close()
    return v_arr


#=========================
# embeddings
#=========================
class Embed:
    """ Embedding membrane to 2d by LTSA.

    Examples:
        embed = Embed().fit(zyx, step=5, n_neigh=20)
        emb1, emb2 = embed.transform(zyx_new)
    """
    def __init__(self, ):
        """ Initialization.
        """
        self.mf = None
    
    def fit(self, zyx, step=5, n_neigh=20):
        """ Fit points.

        Args:
            zyx (np.ndarray): Points on membrane with shape=(npts,3). Assumed sorted for stepping.
            step (int): Sample points with this step size. If too dense, then result looks not good.
            n_neigh (int): The number of neighbors for LTSA.

        Returns:
            self (Embed): Self after fitting manifold.LocallyLinearEmbedding.
        """
        # downsample points: step in xy, step in z
        # zyx is too dense, downsampling is less computationally expensive and gives good global shape
        zyx_sample = []
        for z in np.unique(zyx[:, 0])[::step]:
            zyx_sample.append(zyx[zyx[:, 0] == z][::step])
        zyx_sample = np.concatenate(zyx_sample, axis=0)

        # adjust n_neigh to a valid value
        n_sample = len(zyx_sample)
        n_neigh = n_neigh if n_neigh < n_sample else n_sample-1

        # LTSA
        self.mf = manifold.LocallyLinearEmbedding(
            n_neighbors=n_neigh, n_components=2,
            method="ltsa"
        )
        self.mf.fit(zyx_sample)
        return self
    
    def transform(self, zyx):
        """ Transform new points.

        Args:
            zyx (np.ndarray): Points to be transformed.

        Returns:
            emb1, emb2 (np.ndarray): Embedding points along axis 1,2.
        """
        emb = self.mf.transform(zyx)
        emb1, emb2 = emb[:, 0], emb[:, 1]
        return emb1, emb2

    def fit_transform(self, zyx, step=5, n_neigh=20):
        """ Combines fit and transform.

        Args:
            See args in self.fit.

        Returns:
            emb1, emb2 (np.ndarray): Embedding points along axis 1,2.
        """
        self.fit(zyx, step, n_neigh)
        return self.transform(zyx)

class Project:
    """ Projecting membrane along normals
    
    Attributes:
        e2: assigned unit vector.
        e1: e1=cross(e2,mean(nxyz)).
    
    Examples:

    """
    def __init__(self, e2=(0,0,1)):
        """ Initialization.

        Args:
            e2 (np.ndarray or tuple): Unit vector along e2 in order [x,y,z].
        """
        # convert e2 to xyz order
        self.e2 = np.asarray(e2)

        # init other variables
        self.e1 = None
        self.xyz_center = None
        self.nxyz_avg = None

    def fit(self, xyz, nxyz):
        """ Fit.

        Args:
            xyz, nxyz (np.ndarray): Points and their normals.

        Returns:
            self (Project): Self object, fitted.
        """
        # convert to xyz to be safe
        self.xyz_center = np.mean(xyz, axis=0)
        self.nxyz_avg = np.mean(nxyz, axis=0)

        # calc e1
        e1 = np.cross(self.e2, self.nxyz_avg)
        self.e1 = e1 / np.linalg.norm(e1)
        return self

    def transform(self, xyz):
        """ Transform.

        Args:
            xyz (np.ndarray): Points to be transformed.

        Returns:
            p1, p2 (np.ndarray): Projected coordinates on e1,e2.
        """
        xyz = xyz - self.xyz_center
        p1 = np.dot(xyz, self.e1)
        p2 = np.dot(xyz, self.e2)
        return p1, p2

    def fit_transform(self, xyz, nxyz):
        """ Combines fit and transform.

        Args:
            See args for self.fit.

        Returns:
            p1, p2 (np.ndarray): Projected coordinates on e1,e2.
        """
        self.fit(xyz, nxyz)
        return self.transform(xyz)
