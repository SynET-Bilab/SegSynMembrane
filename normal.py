import collections
import multiprocessing.dummy
import numpy as np
from scipy import spatial
from etsynseg import utils, hessian

__all__ = [
    "align_normal", "surface_normal_one", "surface_normal"
]

def align_normal(coords, normals, coord_ref, radius=3):
    """ align normal to the same direction
    direction of normals:
        the point nearest coord_ref: from ref to point
        other points: propagated from the nearest
    :param coords: coordinates, xyz or zyx, shape=(npts,ndim)
    :param normals: same ordering as coord, shape=(npts,ndim)
    :param coord_ref: reference point
    :param radius: radius for neighbor-search (used 1-norm)
    :return: normals
        normals: aligned normals
    """
    # setup
    normals = np.copy(normals)
    kdtree = spatial.KDTree(coords)
    dists = np.linalg.norm(coords-coord_ref, axis=1)

    signs = np.ones(len(coords), dtype=int)
    unaligned = np.ones(len(coords), dtype=bool)
    tovisit = collections.deque()

    # visit all points
    while np.any(unaligned):
        # start from the nearest unaligned point
        # align its normal to the connecting line
        # and use the normal as the reference
        imin = np.argmin(dists)
        dot_imin = np.dot(normals[imin], coords[imin]-coord_ref)
        normals[imin] *= np.sign(dot_imin)
        unaligned[imin] = False
        tovisit.append(imin)

        # iterate over connected points
        while tovisit:
            i = tovisit.popleft()

            # find unaligned
            neighbors_i = np.asarray(kdtree.query_ball_point(
                coords[i], r=radius, p=1))
            news_i = neighbors_i[unaligned[neighbors_i]]

            # align new points with the current
            if len(news_i) > 0:
                dots_i = np.sum(normals[news_i] * normals[i], axis=1)
                signs[news_i] = signs[i] * np.sign(dots_i)
                unaligned[news_i] = False
                tovisit.extend(list(news_i))

    # reorient normals
    normals[signs < 0] *= -1
    return normals

def surface_normal_one(H_mat):
    """ calculate surface normal at one location, via hessian
    :param H_mat: 3*3 hessian matrix at the location
    :return: normal
        normal: normal vector, [x,y,z]
    """
    evals, evecs = np.linalg.eigh(H_mat)
    # max eig by abs
    imax = np.argmax(np.abs(evals))
    normal = evecs[:, imax]
    return normal

def surface_normal(I, sigma, zyx_ref, pos=None):
    """ calculate surface normal at nonzeros of I
    :param I: image, shape=(nz,ny,nx)
    :param sigma: gaussian smoothing
    :param zxy_ref: [z,y,x] of a reference point for inside
    :param pos: positions of points, for customized ordering or subsetting
    :return: xyz, normals
        normals: normal vectors pointing outwards, [[nx1,ny1,nz1],...]
    """
    # setups
    if pos is None:
        pos = np.nonzero(I)
    npts = len(pos[0])
    
    # coordinates and hessian
    xyz = utils.reverse_coord(np.transpose(pos))
    H_mats = hessian.symmetric_image(hessian.hessian3d(I, sigma))[pos]

    # normal
    normals = np.zeros((npts, 3))

    def calc_one(i):
        normals[i] = surface_normal_one(H_mats[i])

    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(npts))
    pool.close()

    # direction from ref point to the center of image
    xyz_ref = np.asarray(zyx_ref)[::-1]
    normals = align_normal(xyz, normals, xyz_ref, radius=3)

    return xyz, normals
