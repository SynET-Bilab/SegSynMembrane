import numpy as np
import scipy as sp
from sklearn import manifold
from etsynseg import utils

__all__ = [
    "interpolate_dist", "interpolate_avg",
    "Embed", "Project"
]

#=========================
# membranogram
#=========================

def interpolate_dist(zyx, nzyx, dist, I):
    """ get membranogram by interpolating image I at locations of zyx+nzyx*dist
    :param zyx, nzyx: coordinates of membranes and their normals
    :param dist: scalar, distance from membrane
    :param I: image, shape=(nz,ny,nx) 
    :return: zyx_d, v_d
        zyx_d: coordinates at dist from membrane
        v_d: values for points in zyx_d
    """
    zyx_d = zyx+nzyx*dist
    v_d = sp.ndimage.map_coordinates(
        I, zyx_d.T, order=3
    )
    return zyx_d, v_d

def interpolate_avg(zyx, nzyx, dist_arr, I):
    """ get averaged membranogram
    :param zyx, nzyx: coordinates of membranes and their normals
    :param dist_arr: array of distances from membrane
    :param I: image, shape=(nz,ny,nx) 
    :return: zyx_d, v_d
        zyx_d: coordinates at averaged dist from membrane
        v_d: averaged values
    """
    # init empty arrays
    zyx_d = np.zeros(zyx.shape, dtype=float)
    v_d = np.zeros(zyx.shape[0], dtype=float)

    # accumulate
    for dist in dist_arr:
        zyx_d_i, v_d_i = interpolate_dist(
            zyx, nzyx, dist, I
        )
        zyx_d += zyx_d_i
        v_d += v_d_i
    
    # average
    n = len(dist_arr)
    zyx_d /= n
    v_d /= n
    return zyx_d, v_d


#=========================
# embeddings
#=========================
class Embed:
    """ embedding membrane to 2d by LTSA
    """
    def __init__(self, step=5, n_neigh=20):
        """ init
        :param step: sample zyx with this step size
        :param n_neigh: number of neighbors for LTSA
        """
        self.step = step
        self.n_neigh = n_neigh
        self.mf = None
    
    def fit(self, zyx):
        """ fit
        :param zyx: points on membrane (assumed sorted)
        :return: self
        """
        # downsample points: step in xy, step in z
        # zyx is too dense, downsampling is less computationally expensive and gives good global shape
        zyx_sample = []
        for z in np.unique(zyx[:, 0])[::self.step]:
            zyx_sample.append(zyx[zyx[:, 0] == z][::self.step])
        zyx_sample = np.concatenate(zyx_sample, axis=0)

        # adjust n_neigh to a valid value
        n_sample = len(zyx_sample)
        n_neigh = self.n_neigh if self.n_neigh < n_sample else n_sample-1

        # LTSA
        self.mf = manifold.LocallyLinearEmbedding(
            n_neighbors=n_neigh, n_components=2,
            method="ltsa"
        )
        self.mf.fit(zyx_sample)
        return self
    
    def transform(self, zyx):
        """ transform
        :param zyx: coord to be transformed
        :return: emb1, emb2
            emb1,emb2: embedded points along axis 1,2
        """
        emb = self.mf.transform(zyx)
        emb1, emb2 = emb[:, 0], emb[:, 1]
        return emb1, emb2

    def fit_transform(self, zyx):
        """ combines fit and transform
        :return: results of transform
        """
        self.fit(zyx)
        return self.transform(zyx)

class Project:
    """ projecting membrane along normals
    """
    def __init__(self, e2=(0, 1., 0)):
        """ init
        :param e2: unit vector, e1=mean(nzyx)
        """
        # setup e2
        self.e2 = np.asarray(e2) if e2 is not None else np.array([0, 1., 0])

        # init other variables
        self.e1 = None
        self.xyz_center = None
        self.nxyz_avg = None

    def fit(self, zyx, nzyx):
        """ fit
        :param zyx, nzyx: coordinates of membranes and their normals
        :return: self
        """
        # convert to xyz to be safe
        self.xyz_center = np.mean(zyx, axis=0)[::-1]
        self.nxyz_avg = np.mean(nzyx, axis=0)[::-1]

        # calc e1
        e1 = np.cross(self.e2, self.nxyz_avg)
        self.e1 = e1 / np.linalg.norm(e1)
        return self

    def transform(self, zyx):
        """ transform
        :param zyx: coord to be transformed
        :return: p1, p2
            p1,p2: projected coordinates on e1,e2
        """
        xyz = utils.reverse_coord(zyx) - self.xyz_center
        p1 = np.dot(xyz, self.e1)
        p2 = np.dot(xyz, self.e2)
        return p1, p2

    def fit_transform(self, zyx, nzyx):
        """ combines fit and transform
        :return: results of transform
        """
        self.fit(zyx, nzyx)
        return self.transform(zyx)
