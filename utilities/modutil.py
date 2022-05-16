""" Utilities for dealing with imod model.
"""

import numpy as np
import scipy as sp
import pandas as pd
# import skimage
import tslearn.metrics
from etsynseg import pcdutil, bspline

__all__ = [
    # conversions
    "points2model",
    # contour interpolation
    "match_two_contours", "interpolate_contours_alongz", "interpolate_contours_alongxy",
    # regions from contour
    "region_surround_contour_slice", "region_surround_contour",
    # mask
    "mask_from_model"
]

#=========================
# conversions
#=========================

def points2model(zyx_arr, break_contour=None):
    """ Convert points to model DataFrame.
    
    Each element in the point array is one object.
    Points at different z's are assigned to different contours.
    Points in the same xy plane are assigned to the same contour if break_contour=None,
    or are broken into different contours if their distance is > break_contour(float).

    Args:
        zyx_arr (list of np.ndarray): Array of points.
            zyx_arr = [zyx_0,zyx_1,...],
            zyx_i=[[z0,y0,x0],...], with shape=(npts_i,3).
        break_contour (float or None): When to break points into different contours.

    Returns:
        model (pd.DataFrame): Dataframe object for the points, with columns=[object,contour,x,y,z].
    """
    cols = ["object", "contour", "x", "y", "z"]
    dtypes = [int, int, float, float, float]
    data = []

    # iterate over objects
    for i_obj, zyx_obj in enumerate(zyx_arr):
        z_obj = zyx_obj[:, 0]
        ct_prev = 0

        # iterate over xy planes
        for z in np.unique(z_obj):
            xyz_z = pcdutil.reverse_coord(zyx_obj[z_obj == z])
            ones = np.ones((len(xyz_z), 1), dtype=int)

            # assign all points into one contour
            if break_contour is None:
                ct_z = ones * (ct_prev+1)
                ct_prev += 1
            
            # break points into different contours by distance
            else:
                # distance between points
                dist_z = np.linalg.norm(np.diff(xyz_z, axis=0), axis=1)
                # positions where the gap between points is large
                pos_gap = np.nonzero(dist_z > break_contour)[0] + 1
                ct_z = np.zeros(len(xyz_z), dtype=int)
                ct_z[pos_gap] = 1
                # cumulate gaps to get contour indexes starting from 0
                ct_z = np.cumsum(ct_z)
                # add prev contour index to get absolute ones
                ct_z += ct_prev + 1
                # update prev contour index
                ct_prev = ct_z[-1]
                # reshape
                ct_z = ct_z.reshape((-1, 1))

            # collect into data
            data_z = np.concatenate([(i_obj+1)*ones, ct_z, xyz_z], axis=1)
            data.append(data_z)

    data = np.concatenate(data)

    # make dataframe
    model = pd.DataFrame({
        cols[i]: pd.Series(data[:, i], dtype=dtypes[i])
        for i in range(5)
    })
    return model


#=========================
# contour interpolation
#=========================

def match_two_contours(pts1, pts2, closed=False):
    """ Match the ordering of points in contour2 to that of contour1.

    Points in the contours are assumed sorted.
    Dynamic time warping (dtw) is used to match the points and measure the loss.
    For open contours, contour1 is compared with the forward and reverse ordering of contour2.
    For closed contours, contour1 is compared with the forward and reverse ordering of contour2, as well as its rolls.
    The ordering of contour2 with the lowest loss is adopted.
    
    Args:
        pts1, pts2 (np.ndarray): Points in contour1,2, each with shape=(npts,dim).
        closed (bool): Whether the contour is closed or not.

    Returns:
        pts2_best (np.ndarray): Same points as in pts2, but in the best order that matches pts1.
        path_best (list of 2-tuples): Pairing between contours, [(i10,i20),(i11,i21),...], where pts1[i1j] and pts2_best[i2j] are matched pairs.
    """
    # generate candidate reordered pts2
    pts2_rev = pts2[::-1]
    # if open contour: original + reversed
    if not closed:
        pts2_arr = [pts2, pts2_rev]
    # if close contour: original + rolls + reversed + rolls of reversed
    else:
        n2 = len(pts2)
        pts2_arr = (
            [np.roll(pts2, i, axis=0) for i in range(n2)]
            + [np.roll(pts2_rev, i, axis=0) for i in range(n2)]
        )

    # calc dtw for each candidate reordered pts2
    path_arr = []
    loss_arr = []
    for pts2_i in pts2_arr:
        path_i, loss_i = tslearn.metrics.dtw_path(pts1, pts2_i)
        path_arr.append(path_i)
        loss_arr.append(loss_i)

    # select the best order
    i_best = np.argmin(loss_arr)
    pts2_best = pts2_arr[i_best]
    path_best = path_arr[i_best]
    return pts2_best, path_best

def interpolate_contours_alongz(zyx, closed=False):
    """ Interpolate contours along z direction.
    
    The input contour is sparsely sampled in z direction.
    The contour in each xy-plane is assumed ordered.
    Contours at different z's are first order-matched using dynamic time warping.
    Interpolation is then performed at the missing z's.
    
    Args:
        zyx (np.ndarray): Points in the contour, with shape=(npts,3).
        closed (bool): Whether the contour is closed or not.

    Returns:
        zyx_interp (np.ndarray): Points in the interpolated contour, with shape=(npts_interp,3).
    """
    zyx_ct = np.round(zyx).astype(int)
    z_uniq = sorted(np.unique(zyx_ct[:, 0]))
    z_ct = zyx_ct[:, 0]
    zyx_dict = {}

    for z1, z2 in zip(z_uniq[:-1], z_uniq[1:]):
        # get correspondence between two given contours
        if z1 not in zyx_dict:
            zyx_dict[z1] = zyx_ct[z_ct == z1]
        zyx1 = zyx_dict[z1]
        zyx2_raw = zyx_ct[z_ct == z2]
        zyx2, path12 = match_two_contours(zyx1, zyx2_raw, closed=closed)
        zyx_dict[z2] = zyx2

        # linear interpolation on intermediate z's
        for zi in range(z1+1, z2):
            zyx_dict[zi] = np.array([
                ((z2-zi)*zyx1[p[0]] + (zi-z1)*zyx2[p[1]])/(z2-z1)
                for p in path12
            ])

    # round to int
    zyx_interp = np.concatenate([
        zyx_dict[zi]
        for zi in range(z_uniq[0], z_uniq[-1]+1)
    ], axis=0)

    return zyx_interp

def interpolate_contours_alongxy(zyx, degree=2):
    """ Interpolate open contours in xy-planes.
    
    Args:
        zyx (np.ndarray): Points in the contour, with shape=(npts,3).
        degree (int): The degree for bspline interpolation.

    Returns:
        zyx_interp (np.ndarray): Points in the interpolated contour, with shape=(npts_interp,3).
    """
    # deduplicate
    zyx = pcdutil.points_deduplicate(zyx)
    
    # iterate over z's
    z_arr = zyx[:, 0]
    zyx_interp = []
    for z_i in np.unique(z_arr):
        # fit yx points
        yx_i = zyx[z_arr==z_i][:, 1:]
        fit = bspline.Curve(degree).interpolate(yx_i)

        # evaluate at dense parameters, then deduplicate
        n_eval = int(2*pcdutil.wireframe_length(yx_i))
        yx_interp_i = fit(np.linspace(0, 1, n_eval))
        yx_interp_i = pcdutil.points_deduplicate(yx_interp_i)

        # stack yx with z
        z_ones_i = z_i*np.ones((len(yx_interp_i), 1))
        zyx_interp_i = np.concatenate([
            z_ones_i, yx_interp_i
        ], axis=1)
        
        zyx_interp.append(zyx_interp_i)
    
    zyx_interp = np.concatenate(zyx_interp, axis=0)
    return zyx_interp


#=========================
# regions from contour
#=========================

def region_surround_contour_slice(yx, nyx, shape, probe_end=10):
    """ Generate regions surrounding a 2d open contour, e.g. a line.

    Args:
        yx (np.ndarray): 2d points, with shape=(npts,2), arranged as [[y0,x0],...].
        nyx (np.ndarray): 2d normals of the points, with shape=(npts,2), arranged as [[ny0,nx0],...].
        shape (2-tuple): Shape of the 2d image, (ny,nx).
        probe_end (int): Direction for outside is aligned with yx[0]-yx[probe_end] (likewise for the other end).

    Returns:
        dist (np.ndarray): Shape=shape. Distance of each pixel to its closest point in yx.
        dot_norm (np.ndarray): Shape=shape. Dot product between each pixel's shift from the closest yx-point and the normal of that point.
        mask_in (np.ndarray): Shape=shape. False for points that extend beyond the endpoints.
    """
    # setup points
    pos_yx = tuple(yx.astype(int).T)
    Y, X = np.mgrid[:shape[0], :shape[1]]

    # assign values to positions
    Iny = np.zeros(shape)
    Iny[pos_yx] = nyx[:, 0]
    Inx = np.zeros(shape)
    Inx[pos_yx] = nyx[:, 1]

    # distance
    # generate background (bg) image with 0's at point yx's
    Iyx = np.ones(shape)
    Iyx[pos_yx] = 0
    # dist(ny,nx): each pixel's distance to bg
    # idx(2,ny,nx): index of closest bg point, which is also the coordinate
    dist, (idxY, idxX) = sp.ndimage.distance_transform_edt(
        Iyx, return_indices=True
    )

    # dot product between shift and normal
    # shift from the closest bg point for each pixel
    dY = Y-idxY
    dX = X-idxX
    # normals of the closest bg point for each pixel
    nY = Iny[idxY, idxX]
    nX = Inx[idxY, idxX]
    # dot product
    dot_norm = dY*nY + dX*nX

    # mask out points that extend beyond the endpoints
    def gen_mask_inside(i_end, i_probe):
        """ Generate masks to exclude points beyond the endpoint.

        Args:
            i_end (int): Index of the endpoint.
            i_probe (int): Index of the probe point, for determining the direction towards outside.

        Returns:
            mask_in (np.ndarray): Shape=shape, True for pixels inside.
        """
        # each pixel's shift relative to the endpoint
        dYend = Y - yx[i_end, 0]
        dXend = X - yx[i_end, 1]
        # direction towards outside and perpendicular to normal
        d_probe = yx[i_end] - yx[i_probe]
        nyx_end = nyx[i_end] / np.linalg.norm(nyx[i_end])
        d_out = d_probe - np.dot(d_probe, nyx_end)*nyx_end
        # pixels inside determined by the dot product
        mask_in = (dYend*d_out[0] + dXend*d_out[1]) <= 0
        return mask_in

    # setup probe
    probe_end = min(probe_end, len(yx)-1)
    # mask for both ends
    mask_in = (
        gen_mask_inside(0, probe_end)
        & gen_mask_inside(-1, -1-probe_end)
    )

    return dist, dot_norm, mask_in

def region_surround_contour(zyx, nzyx, width, cut_end=True):
    """ Generate regions surrounding a 3d open contour, e.g. a surface.

    Regions are extended from the surface along local normal directions (projected to xy-plane, no extensions along z).
    Different widths can be set for regions in the normal direction and its opposite.
    Regions can be round-headed at the endpoints, or be cut beyond the normals of the endpoints.

    Args:
        zyx (np.ndarray): 3d points, with shape=(npts,3), arranged as [[z0,y0,x0],...].
        nzyx (np.ndarray): 3d normals of the points, with shape=(npts,3), arranged as [[nz0,ny0,nx0],...].
        width (float or 2-tuple): Width to extend in normal's plus and minus direction, (width+,width-), or a float for equal widths.
        cut_end (bool): Whether the region near the endpoints is round-headed (False) or will be cut beyond their normals (True).

    Returns:
        zyx_plus (np.ndarray): 3d points of the mask in the normal+ direction, with shape=(npts_plus,3).
        zyx_minus (np.ndarray): 3d points of the mask in the normal- direction, with shape=(npts_minus,3).
    """
    # setup width=(width_plus,width_minus)
    if isinstance(width, (int, float)):
        width = (width, width)

    # calculate yx ranges for later clipping
    zyx = np.round(zyx).astype(int)
    yx_low, _, yx_shape = pcdutil.points_range(zyx[:, 1:], margin=int(max(width))+1)

    # generate regions for each slice
    zyx_plus = []
    zyx_minus = []
    for z in sorted(np.unique(zyx[:, 0])):
        # extract data at z
        # subtract the shift to reduce size
        mask_z = zyx[:, 0] == z
        yx_clip = zyx[mask_z][:, 1:] - yx_low
        nyx = nzyx[mask_z][:, 1:]

        # get 2d masks at each pixel
        dist, dot_norm, mask_in = region_surround_contour_slice(
            yx_clip, nyx, yx_shape
        )

        # extract regions from masks
        mask_plus = (dot_norm > 0) & (dist <= width[0])
        mask_minus = (dot_norm < 0) & (dist <= width[1])
        if cut_end:
            mask_plus = mask_plus & mask_in
            mask_minus = mask_minus & mask_in

        # convert pixels to points, add back the shift
        yx_plus = pcdutil.pixels2points(mask_plus) + yx_low
        yx_minus = pcdutil.pixels2points(mask_minus) + yx_low

        # concat with z
        zyx_plus_z = np.concatenate([
            z*np.ones((len(yx_plus), 1)), yx_plus
        ], axis=1)
        zyx_minus_z = np.concatenate([
            z*np.ones((len(yx_minus), 1)), yx_minus
        ], axis=1)

        zyx_plus.append(zyx_plus_z)
        zyx_minus.append(zyx_minus_z)

    zyx_plus = np.concatenate(zyx_plus, axis=0).astype(int)
    zyx_minus = np.concatenate(zyx_minus, axis=0).astype(int)

    return zyx_plus, zyx_minus


#=========================
# model to mask
#=========================

def mask_from_model(zyx_mod, width, normal_ref=None, interp_degree=2, cut_end=True):
    """ Convert imod model (sparsely-sampled surface) to mask of regions surrounding it.

    First interpolate the model along z and xy, to make dense contours.
    Then generate the region surrounding the contours.

    Args:
        zyx_mod (np.ndarray): Model points, ordered in each z.
        width (float or 2-tuple): Width to extend in normal's plus and minus direction, (width+,width-), or a float for equal widths.
        interp_degree (int): Degree for bspline interpolation in the xy direction.
        normal_ref (np.ndarray or None): Reference point 'inside' for orienting the normals, [z_ref,y_ref,x_ref].
            If None, then generate from pcdutil.normals_gen_ref.
        cut_end (bool): Whether the region near the endpoints is round-headed (False) or will be cut beyond their normals (True).

    Returns:
        zyx (np.ndarray): 3d points of the interpolated normal, with shape=(npts,3).
        zyx_plus (np.ndarray): 3d points of the mask in the normal+ direction, with shape=(npts_plus,3).
        zyx_minus (np.ndarray): 3d points of the mask in the normal- direction, with shape=(npts_minus,3).
        normal_ref (np.ndarray): Reference point 'inside' for orienting the normals.
    """
    zyx = np.round(zyx_mod).astype(int)

    # interpolate along z and xy
    zyx = interpolate_contours_alongz(zyx, closed=False)
    zyx = interpolate_contours_alongxy(zyx, degree=interp_degree)
    zyx = pcdutil.points_deduplicate(zyx)

    # estimate normals
    if normal_ref is None:
        normal_ref = pcdutil.normals_gen_ref(zyx)
    else:
        normal_ref = np.asarray(normal_ref)
    # sigma is set to the range in z: bsplines may not be smooth along z
    normal_sigma = np.ptp(zyx[:, 0])/2
    nzyx = pcdutil.normals_points(
        zyx, sigma=normal_sigma, pt_ref=normal_ref
    )

    # generate regions
    zyx_plus, zyx_minus = region_surround_contour(
        zyx, nzyx, width=width, cut_end=cut_end
    )

    return zyx, zyx_plus, zyx_minus, normal_ref
