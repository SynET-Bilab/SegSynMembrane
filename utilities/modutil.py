""" Utilities for dealing with imod model.
"""

import pathlib
import numpy as np
import scipy as sp
import multiprocessing.dummy
import tslearn.metrics
from etsynseg import pcdutil, bspline, io

__all__ = [
    # contour interpolation
    "match_two_contours", "interpolate_contours_alongz", "interpolate_contours_alongxy",
    # regions from contour
    "region_surround_contour_slice", "region_surround_contour",
    # read tomo model
    "regions_from_guide", "read_tomo_model", "read_tomo_clip"
]

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

def region_surround_contour_slice(yx, nyx, shape, probe_end=5):
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

    # assign normals to positions
    Iny = np.zeros(shape)
    Iny[pos_yx] = nyx[:, 0]
    Inx = np.zeros(shape)
    Inx[pos_yx] = nyx[:, 1]

    # distance
    # generate background (bg) image with 0's at point yx's
    Iyx = np.ones(shape)
    Iyx[pos_yx] = 0
    # dist(ny,nx): each pixel's distance to bg
    # idxY, idxX (ny,nx): index of closest bg point, which is also the coordinate
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
        # mask_out by shift direction
        # each pixel's shift relative to the endpoint
        dYend = Y - yx[i_end, 0]
        dXend = X - yx[i_end, 1]
        # direction towards outside and perpendicular to normal
        d_probe = yx[i_end] - yx[i_probe]
        nyx_end = nyx[i_end] / np.linalg.norm(nyx[i_end])
        d_out = d_probe - np.dot(d_probe, nyx_end)*nyx_end
        # pixels outside determined by the dot product
        mask_dir = (dYend*d_out[0] + dXend*d_out[1]) >= 0

        # mask_out by closest point
        # only consider points whose closest point is in [i_end, i_probe)
        mask_closest = np.zeros(shape, dtype=bool)
        for i in range(i_end, i_probe, np.sign(i_probe-i_end)):
            mask_closest[(idxY==yx[i, 0])&(idxX==yx[i, 1])] = True
        
        # combine masks, reverse to get inside
        mask_out = mask_dir & mask_closest
        mask_in = ~mask_out
        return mask_in

    # setup probe
    probe_end = min(probe_end, len(yx)-1)
    # mask for both ends
    mask_in = (
        gen_mask_inside(0, probe_end)
        & gen_mask_inside(-1, -1-probe_end)
    )

    return dist, dot_norm, mask_in

def region_surround_contour(zyx, nzyx, extend, cut_end=True):
    """ Generate regions surrounding a 3d open contour, e.g. a surface.

    Regions are extended from the surface along local normal directions (projected to xy-plane, no extensions along z).
    Different extensions can be set for regions in the normal direction and its opposite.
    Regions can be round-headed at the endpoints, or be cut beyond the normals of the endpoints.

    Args:
        zyx (np.ndarray): 3d points, with shape=(npts,3), arranged as [[z0,y0,x0],...].
        nzyx (np.ndarray): 3d normals of the points, with shape=(npts,3), arranged as [[nz0,ny0,nx0],...].
        extend (float or 2-tuple): Width to extend in normal's plus and minus direction, (extend+,extend-), or a float for equal extensions.
        cut_end (bool): Whether the region near the endpoints is round-headed (False) or will be cut beyond their normals (True).

    Returns:
        zyx_plus (np.ndarray): 3d points of the mask in the normal+ direction, with shape=(npts_plus,3).
        zyx_minus (np.ndarray): 3d points of the mask in the normal- direction, with shape=(npts_minus,3).
    """
    # setup extend=(extend_plus,extend_minus)
    if isinstance(extend, (int, float)):
        extend = (extend, extend)

    # calculate yx ranges for later clipping
    zyx = np.round(zyx).astype(int)
    extend_max = int(max(extend))+1
    clip_low, _, clip_shape = pcdutil.points_range(
        zyx,
        margin=(0, extend_max, extend_max),
        clip_neg=True
    )
    zyx_clip = zyx - clip_low

    # generate regions for one slice
    mask_plus = np.zeros(clip_shape, dtype=bool)
    mask_minus = np.zeros(clip_shape, dtype=bool)
    def calc_one(iz):
        # extract data at z
        mask_z = zyx_clip[:, 0] == iz
        yx_clip = zyx_clip[mask_z][:, 1:]
        nyx = nzyx[mask_z][:, 1:]
        # get 2d masks at each pixel
        dist, dot_norm, mask_in = region_surround_contour_slice(
            yx_clip, nyx, clip_shape[1:]
        )
        # extract regions from masks
        mask_plus_z = (dot_norm > 0) & (dist <= extend[0])
        mask_minus_z = (dot_norm < 0) & (dist <= extend[1])
        if cut_end:
            mask_plus_z = mask_plus_z & mask_in
            mask_minus_z = mask_minus_z & mask_in
        # assignment
        mask_plus[iz] = mask_plus_z
        mask_minus[iz] = mask_minus_z
    
    # generate regions for all slices
    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(clip_shape[0]))
    pool.close()

    # convert masks to points
    zyx_plus = pcdutil.pixels2points(mask_plus) + clip_low
    zyx_minus = pcdutil.pixels2points(mask_minus) + clip_low
    return zyx_plus, zyx_minus


#=========================
# read tomo, model
#=========================

def regions_from_guide(guide_mod, extend, normal_ref=None, interp_degree=2, cut_end=True):
    """ Convert imod model with guiding lines to regions surrounding it.

    First interpolate the model along z and xy, to make dense contours.
    Then generate the region surrounding the contours.
    Regions are represented by points. If unclipped points are converted to image, it may take much space.

    Args:
        guide_mod (np.ndarray): Model points of guiding lines, ordered in each z.
        extend (float or 2-tuple): Width to extend in normal's plus and minus direction, (extend+,extend-), or a float for equal extensions.
        interp_degree (int): Degree for bspline interpolation in the xy direction.
        normal_ref (np.ndarray or None): Reference point 'inside' for orienting the normals, [z_ref,y_ref,x_ref].
            If None, then generate from pcdutil.normals_gen_ref.
        cut_end (bool): Whether the region near the endpoints is round-headed (False) or will be cut beyond their normals (True).

    Returns:
        guide (np.ndarray): Points of the interpolated guiding lines, with shape=(npts,3).
        bound_plus (np.ndarray): Points of the bounding region in the normal+ direction, with shape=(npts_plus,3).
        bound_minus (np.ndarray): Points of the bounding region in the normal- direction, with shape=(npts_minus,3).
        normal_ref (np.ndarray): Reference point 'inside' for orienting the normals.
    """
    # regulate guiding lines
    # remove slices where the number of points <= 1
    guide_raw = pcdutil.points_deduplicate(guide_mod)
    guide = []
    for z in np.unique(guide_raw[:, 0]):
        guide_raw_z = guide_raw[guide_raw[:, 0]==z]
        if len(guide_raw_z) > 1:
            guide.append(guide_raw_z)
    guide = np.concatenate(guide, axis=0)

    # guiding lines interpolation along z and xy
    guide = interpolate_contours_alongz(guide, closed=False)
    guide = interpolate_contours_alongxy(guide, degree=interp_degree)
    guide = pcdutil.points_deduplicate(guide)

    # estimate normals
    if normal_ref is None:
        normal_ref = pcdutil.normals_gen_ref(guide)
    else:
        normal_ref = np.asarray(normal_ref)
    # sigma is set to the range in z: bsplines may not be smooth along z
    normal_sigma = np.ptp(guide[:, 0])/2
    normals = pcdutil.normals_points(
        guide, sigma=normal_sigma, pt_ref=normal_ref
    )

    # generate regions
    bound_plus, bound_minus = region_surround_contour(
        guide, normals, extend=extend, cut_end=cut_end
    )

    return guide, bound_plus, bound_minus, normal_ref

def read_tomo_model(tomo_file, model_file, extend_nm, pixel_nm=None, interp_degree=2, raise_noref=False):
    """ Read tomo and model.

    Read model: object 1 for guiding lines, object 2 for reference point.
    Generate region mask surrounding the guiding lines.
    Decide clip range from the region mask.
    Read tomo, clip.

    Args:
        tomo_file (str): Filename of tomo mrc.
        model_file (str): Filename of imod model.
        extend_nm (float): Extend from guiding lines by this value (in nm) to get the bound.
        pixel_nm (float): Pixel size in nm. If None then read from tomo.
        interp_degree (int): Degree of bspline interpolation of the model.
            2 for most cases.
            1 for finely-drawn model to preserve the original contours.
        raise_noref (bool): Whether to raise error if the reference point is not found in model object 2.

    Returns:
        tomod (dict): Tomo, model, and relevant info. See below for fields in the dict.
            I: clipped tomo
            shape: I.shape
            pixel_nm: pixel size in nm
            model: model DataFrame, in the original coordinates
            clip_low: [z,y,x] at the lower corner for clipping
            guide: interpolated guiding lines, in points, with shape=(npts_guide,3)
            bound, bound_plus, bound_minus: binary masks extended from guide towards both/normal+/normal- directions, shape=I.shape.
            normal_ref: reference point inside for normal orientation
    """
    # check file existence
    if not pathlib.Path(tomo_file).is_file():
        raise FileNotFoundError(f"Tomo file not found: {tomo_file}")
    if not pathlib.Path(model_file).is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # read tomo
    I, pixel_A = io.read_tomo(tomo_file, mode="mmap")
    # get pixel size in nm
    if pixel_nm is None:
        pixel_nm = pixel_A / 10

    # read model
    model = io.read_model(model_file)
    # object: guiding lines
    if 1 in model["object"].values:
        model_guide = model[model["object"] == 1][['z', 'y', 'x']].values
    else:
        raise ValueError("The object for guiding lines (id=1) is not found in the model file.")
    # object: normal ref point
    if 2 in model["object"].values:
        normal_ref = model[model["object"] == 2][['z', 'y', 'x']].values[0]
    elif raise_noref:
        raise ValueError("The object for reference point (id=2) is not found in the model file.")
    else:
        normal_ref = None

    # generate bounding regions (in points) from guiding line
    guide, boundpts_plus, boundpts_minus, normal_ref = regions_from_guide(
        model_guide,
        extend=extend_nm/pixel_nm,
        normal_ref=normal_ref,
        interp_degree=interp_degree,
        cut_end=True
    )
    # restrict guide and bound to >=0
    guide = np.clip(guide, 0, np.inf).astype(int)
    boundpts_minus = np.clip(boundpts_minus, 0, np.inf).astype(int)
    boundpts_plus = np.clip(boundpts_plus, 0, np.inf).astype(int)
    # combine all parts of the bound
    boundpts = np.concatenate([guide, boundpts_plus, boundpts_minus], axis=0)

    # get clip range and raw shape from the mask
    clip_low, _, shape = pcdutil.points_range(boundpts, margin=0, clip_neg=True)

    # clip coordinates
    guide -= clip_low
    boundpts_plus -= clip_low
    boundpts_minus -= clip_low
    boundpts -= clip_low
    normal_ref -= clip_low

    # clip tomo
    sub = tuple(slice(ci, ci+si) for ci, si in zip(clip_low, shape))
    I = I[sub]
    shape = I.shape

    # convert bounds from points to image
    bound = pcdutil.points2pixels(boundpts, shape, dtype=bool)
    bound_plus = pcdutil.points2pixels(boundpts_plus, shape, dtype=bool)
    bound_minus = pcdutil.points2pixels(boundpts_minus, shape, dtype=bool)

    # save parameters and results
    tomod = dict(
        I=I,
        shape=shape,
        pixel_nm=pixel_nm,
        model=model,
        clip_low=clip_low,
        bound=bound,
        normal_ref=normal_ref,
        guide=guide,
        bound_plus=bound_plus,
        bound_minus=bound_minus
    )
    return tomod

def read_tomo_clip(tomo_file, zyx, margin_nm=0, pixel_nm=None):
    """ Read tomo, clip to the range of points (+margin).

    Args:
        tomo_file (str): Filename of tomo mrc.
        zyx (np.ndarray): Points with shape=(npts,3), each element is [zi,yi,xi].
        margin_nm (float): Margin (in nm) of clipped tomo from the range of points.
        pixel_nm (float): Pixel size in nm. If None then read from tomo.

    Returns:
        I (np.ndarray): Clipped tomo, shape=[nz,ny,nx].
        clip_low (np.ndarray): [z,y,x] at the lower corner for clipping.
            zyx-clip_low gives coordinates in the clipped tomo.
        pixel_nm (float): Pixel size in nm.
    """
    # read tomo
    I, pixel_A = io.read_tomo(tomo_file, mode="mmap")
    # get pixel size in nm
    if pixel_nm is None:
        pixel_nm = pixel_A / 10

    # get clip range
    margin = margin_nm / pixel_nm
    clip_low, clip_high, _ = pcdutil.points_range(zyx, margin=margin, clip_neg=True)
    # # restrict to nonnegative values
    # clip_low = np.clip(clip_low, 0, np.inf).astype(int)

    # clip tomo
    sub = tuple(slice(low, high+1) for low, high in zip(clip_low, clip_high))
    I = I[sub]

    return I, clip_low, pixel_nm
