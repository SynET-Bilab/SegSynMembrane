""" utils: common utilities
"""
import numpy as np
import pandas as pd
import scipy as sp
import skimage
import open3d as o3d

__all__ = [
    # basics
    "zscore_image", "minmax_image", "negate_image", "gaussian",
    # orientation
    "rotate_orient", "absdiff_orient",
    # points, image
    "voxels_to_points", "points_to_voxels", "reverse_coord", "mask_to_contour", "points_to_pointcloud",
    # sparse
    "sparsify3d", "densify3d",
    # segments
    "extract_connected", "stats_per_label", "filter_connected_xy", "filter_connected_dz",
    # grid helpers
    "spans_xy", "wireframe_lengths",
]

#=========================
# basic processing
#=========================

def zscore_image(I):
    """ zscore image I """
    z = (I-np.mean(I))/np.std(I)
    return z

def minmax_image(I, qrange=(0, 1), vrange=(0, 1)):
    """ minmax-scale of image I
    :param qrange: clip I by quantile range
    :param vrange: target range of values
    """
    # clip I by quantiles, set by qrange
    I_min = np.quantile(I, qrange[0])
    I_max = np.quantile(I, qrange[1])
    I_clip = np.clip(I, I_min, I_max)

    # scale to
    I_scaled = vrange[0] + (I_clip-I_min)/(I_max-I_min)*(vrange[1]-vrange[0])
    return I_scaled

def negate_image(I):
    """ switch between white and dark foreground, zscore->negate
    """
    std = np.std(I)
    if std > 0:
        return -(I-np.mean(I))/std
    else:
        return np.zeros_like(I)

def gaussian(I, sigma):
    """ gaussian smoothing, a wrapper of skimage.filters.gaussian
    :param sigma: if sigma=0, return I
    """
    if sigma == 0:
        return I
    else:
        return skimage.filters.gaussian(I, sigma, mode="nearest")

#=========================
# orientation tools
#=========================

def rotate_orient(O):
    """ rotate orientation by pi/2, then mod pi
    :return: mod(O+pi/2, pi)
    """
    return np.mod(O+np.pi/2, np.pi)

def absdiff_orient(O1, O2):
    """ abs diff between O1, O2, converted to (0, pi/2)
    :param O1, O2: orientations, in (-pi/2, pi/2)
    :return: dO
    """
    dO = np.abs(O1-O2)
    dO = np.where(dO<=np.pi/2, dO, np.pi-dO)
    return dO


#=========================
# coordinates tools
#=========================

def voxels_to_points(B):
    """ convert B[z,y,x] to coordinates (z,y,x) of points>0
    :param B: binary image
    :return: coord
        coord: shape=(npts, B.ndim)
    """
    pts = np.argwhere(B)
    return pts

def points_to_voxels(pts, shape):
    """ convert coordinates (y,x) to B[y,x] with 1's on points
    :param pts: yx or zyx
    :param shape: (ny,nx) or (nz,ny,nx)
    :return: B
    """
    # round to int
    pts = np.round(np.asarray(pts)).astype(int)

    # remove out-of-bound points
    ndim = pts.shape[1]
    mask = np.ones(len(pts), dtype=bool)
    for i in range(ndim):
        mask_i = (pts[:, i]>=0) & (pts[:, i]<=shape[i]-1)
        mask = mask & mask_i
    pts = pts[mask]

    # assign to voxels
    B = np.zeros(shape, dtype=int)
    index = tuple(pts.T)
    B[index] = 1
    return B

def reverse_coord(pts):
    """ convert (y,x) to (x,y)
    :param pts: yx or zyx
    :return: reversed pts
    """
    pts = np.asarray(pts)
    index_rev = np.arange(pts.shape[1])[::-1]
    return pts[:, index_rev]

def mask_to_contour(mask, erode=True):
    """ convert a (connected) binary mask to contour (largest one in each plane)
    :param mask: binary image
    :param erode: whether to erode the image first, to avoid broken contours due to the boundary
    :return: contour
        contour: yx for 2d, zyx for 3d
    """
    def get_largest(contours):
        # find the largest of an array of contours
        sizes = [len(c) for c in contours]
        imax = np.argmax(sizes)
        return contours[imax]

    # 2d case
    if mask.ndim == 2:
        if erode:
            mask = skimage.morphology.binary_erosion(mask)
        contour = get_largest(skimage.measure.find_contours(mask))
    
    # 3d case
    elif mask.ndim == 3:
        contour = []
        for i, mask_i in enumerate(mask):
            if erode:
                mask_i = skimage.morphology.binary_erosion(mask_i)
            yx_i = get_largest(skimage.measure.find_contours(mask_i))
            zyx_i = np.concatenate(
                [i*np.ones((len(yx_i), 1)), yx_i], axis=1)
            contour.append(zyx_i)
        contour = np.concatenate(contour, axis=0)
    return contour

def points_to_pointcloud(pts, normals=None):
    """ create open3d point cloud from points
    :param pts: shape=(npts, ndim) 
    :param normals: shape=(npts, ndim)
    :return: pcd
        pcd: open3d pointcloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

#=========================
# sparse tools
#=========================

def sparsify3d(B):
    """ sparsify 3d image to an array of coo_matrix
    :param B: 3d image
    :return: Bs
    """
    Bs = np.array(
        [sp.sparse.coo_matrix(B[i]) for i in range(len(B))]
    )
    return Bs

def densify3d(Bs):
    """ densify an array of coo_matrix to 3d image
    :param Bs: array of coo_matrix
    :return: B
    """
    B = np.array([Bs[i].todense() for i in range(len(Bs))])
    return B


#=========================
# segment tools
#=========================

def extract_connected(B, n_keep=None, connectivity=2):
    """ extract n_keep largest connected components
    :param B: binary image
    :param n_keep: number of components to keep
    :param connectivity: sense of neighboring, 1(-|) or 2(-|\/)
    :return: yield (count, B_i)
    """
    # label
    L = skimage.measure.label(B, connectivity=connectivity)
    # count
    df = (pd.Series(L[L > 0])
          .value_counts(sort=True, ascending=False)
          .to_frame("count").reset_index()
          )
    # yield
    for item in df.iloc[:n_keep].itertuples():
        B_i = B * (L == item.index)
        yield (item.count, B_i)


def stats_per_label(L, V_arr, name_arr=None, stats="mean",
    qfilter=0.25, min_size=1):
    """ calc statistics for each label in the image
    :param L: image with integer labels on pixels
    :param V_arr: image array with values on pixels
    :param name_arr: names of values, default=["value0","value1",...]
    :param stats: statistics to apply on values, e.g. "mean", "median"
    :param qfilter: filter out labels if any of their stats < qfilter quantile
    :param min_size: min size of segments
    :return: df_stats
        df_stats: columns=["label","count","value0",...], row=each label
    """
    # positions of nonzero pixels, match with values
    pos = np.nonzero(L)
    if name_arr is None:
        name_arr = [f"value{v}" for v in range(len(V_arr))]
    df_px = pd.DataFrame(
        data=np.transpose([L[pos], *[V[pos] for V in V_arr]]),
        columns=["label", *name_arr]
    ).astype({"label": int})

    # count
    df_count = df_px.value_counts("label").to_frame("count")
    df_count = df_count[df_count["count"]>=min_size]
    df_count = df_count.reset_index()

    # group by labels, then stat
    df_stats = df_px.groupby("label").agg(stats)

    # filter
    threshold = df_stats.quantile(qfilter)
    df_stats = df_stats[df_stats>=threshold].dropna(axis=0)
    df_stats = df_stats.reset_index()

    # merge count into stats
    df_stats = pd.merge(df_count, df_stats, on="label", how="inner")
    return df_stats

def filter_connected_xy(B, V_arr, connectivity=2, stats="median",qfilter=0.25, min_size=1):
    """ label by connectivity for each xy-slice, filter out small values
    :param B: binary image
    :param V_arr: array of valued-images
    :param connectivity: used for determining connected segments, 1 or 2
    :param stats: statistics to apply on values
    :param qfilter: filter out labels if any of their stats < qfilter quantile
    :param min_size: min size of segments
    :return: B_filt
        B_filt: filtered binary image
    """
    B_filt = np.zeros_like(B)
    for i in range(B_filt.shape[0]):
        # label by connectivity
        Li = skimage.measure.label(B[i], connectivity=connectivity)

        # stats
        df_stats = stats_per_label(
            Li, [V[i] for V in V_arr],
            stats=stats, qfilter=qfilter, min_size=min_size
        )

        # filter out labels from image
        B_filt[i] = B[i]*np.isin(Li, df_stats["label"])
    return B_filt

def filter_connected_dz(B, dzfilter=1, connectivity=2):
    """ label by connectivity in 3d, filter out dz<dzfilter segments
    :param B: binary image
    :param connectivity: used for determining connected segments, 1 or 2
    :param dzfilter: minimal value of z-range
    :return: B_filt
        B_filt: filtered binary image
    """
    # label
    L = skimage.measure.label(B, connectivity=connectivity)
    # z-value of each pixel
    nz = L.shape[0]
    Z = np.ones(L.shape)*np.arange(nz).reshape((-1,1,1))
    # z-range for each label
    df = stats_per_label(L, [Z], name_arr=["z"],
        stats=(lambda x: np.max(x)-np.min(x)),
        qfilter=0, min_size=dzfilter
    )
    # filter
    mask = np.isin(L, df["label"][df["z"] >= dzfilter])
    B_filt = B * mask
    return B_filt


#=========================
# grid helpers
#=========================

def spans_xy(B):
    """ calculate span in xy for each z
    :param B: image, shape=(nz, ny, nx)
    :return: dydx
        dydx: 2d np.ndarray, [[dy1, dx1],...]
    """
    nz = B.shape[0]
    dydx = np.zeros((nz, 2))
    for iz in range(nz):
        yx = voxels_to_points(B[iz])
        dydx[iz] = np.ptp(yx, axis=0)
    return dydx

def wireframe_lengths(pts_net, axis):
    """ calculate lengths of wireframe along one axis
    :param pts_net: shape=(nu,nv,3)
    :param axis: u - 0, v - 1
    :return: wires
        wires: 1d np.ndarray, [length1, length2, ...]
    """
    # A, B - axes
    # [dz,dy,dx] along A for each B
    diff_zyx = np.diff(pts_net, axis=axis)
    # len of wire segments along A for each B
    segments = np.linalg.norm(diff_zyx, axis=-1)
    # len of wire along A for each B
    wires = np.sum(segments, axis=axis)
    return wires
