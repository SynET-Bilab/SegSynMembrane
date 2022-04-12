import numpy as np
import scipy as sp
from sklearn import manifold
from etsynseg import utils

__all__ = [
    "interpolate_dist", "interpolate_avg",
    "embed", "project"
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

def embed(zyx, step=5, n_neigh=20):
    """ embedding membrane to 2d by LTSA
    :param zyx: points on membrane (assumed sorted)
    :param step: sample zyx with this step size
    :param n_neigh: number of neighbors for LTSA
    :return: emb1, emb2
        emb1,emb2: embedded points along axis 1,2
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
    mf = manifold.LocallyLinearEmbedding(
        n_neighbors=n_neigh, n_components=2,
        method="ltsa"
    )
    mf.fit(zyx_sample)
    emb = mf.transform(zyx)
    return emb[:, 0], emb[:, 1]

def project(zyx, nzyx, e2=(0, 1., 0)):
    """ projecting membrane along normals
    :param zyx, nzyx: coordinates of membranes and their normals
    :param e2: unit vector, e1=mean(nzyx)
    :return: p1, p2
        p1,p2: projected coordinates on e1,e2
    """
    # convert to xyz to be safe
    xyz = utils.reverse_coord(zyx)
    xyz_ct = xyz - np.mean(xyz, axis=0)
    nxyz_avg = np.mean(nzyx, axis=0)[::-1]

    # e2: // +y by default
    # e1: if nzyx//+x, e1//-z
    e2 = np.asarray(e2) if e2 is not None else np.array([0, 1., 0])
    e1 = np.cross(e2, nxyz_avg)
    e1 = e1 / np.linalg.norm(e1)

    # project
    p1 = np.dot(xyz_ct, e1)
    p2 = np.dot(xyz_ct, e2)
    return p1, p2
