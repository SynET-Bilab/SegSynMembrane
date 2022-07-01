""" memsampling: importance sampling on the segmentation
"""
import os
import numpy as np
import scipy as sp
import multiprocessing.dummy
from sklearn import mixture

from etsynseg import pcdutil

__all__ = [
    # box setups
    "gen_box_coords", "gen_box_mapping",
    # box extraction: scalar
    "extract_box_avg",
    # box extraction: tensor
    "extract_box_tensor", "extract_box_3d", "extract_box_2drot",
    # local max
    "localmax_neighbors", "localmax_perslice", "localmax_exclusion",
    # classification
    "classify_2drot_init", "classify_2drot_boxes"
]


#=========================
# box setups
#=========================

def gen_box_coords(box_rn, box_rt, mask_rn=None):
    """ Generate coordinates for pixels in the sampling box.

    For point i, the pixel j of its sampling box locates at:
    zyx[i] + box_coos[0]*nzyx[i] + box_coos[1]*tyx[i] + box_coos[2]*tz[i]

    Args:
        box_rn (2-tuple of float): (min,max) extensions in normal direction.
        box_rt (float): max extension in tangent direction.
        mask_rn (2-tuple of float): Ignore region (min,max) in normal direction.
            E.g. can be used to mask out the membrane.
    
    Returns:
        box_coos (np.ndarray): Coordinates of all pixels in the sampling box.
    """
    # grids
    # convert to int
    box_rn_int = np.round(box_rn).astype(int)
    box_rt_int = int(np.round(box_rt))
    # coordinate-grid in normal and tangent directions
    grid_n, grid_tyx, grid_tz = np.mgrid[
        box_rn_int[0]:box_rn_int[1]+1,
        -box_rt_int:box_rt_int+1,
        -box_rt_int:box_rt_int+1
    ]
    # coordinate-grid in radial direction
    grid_tr = np.sqrt(grid_tyx**2+grid_tz**2)

    # coordinates for each pixel
    mask_coos = (grid_n >= box_rn[0]) & (grid_n <= box_rn[1]) & (grid_tr <= box_rt)
    if mask_rn is not None:
        mask_coos = mask_coos & (grid_n <= mask_rn[0]) & (grid_n >= mask_rn[1])
    box_coos = np.stack([
        gi[mask_coos] for gi in [grid_n, grid_tyx, grid_tz]], axis=1
    )
    return box_coos

def gen_box_mapping(box_coos, box_locs):
    """ Generate the mapping from sampling box to extracting box.

    Inputs: pixel coordinates in the sampling box and corresponding locations in the extracted box.
    Outputs:
        ext_box = np.zeros(ext_shape)
        # values_coos: same len as box_coos
        ext_box[tuple(ext_locs.T)] = mat_coo2loc @ values_coos

    Args:
        box_coos (np.ndarray): The coordinate of each pixel in the sampling box.
            Shape=(npx,dim_box). E.g. [z,y,x].
        box_locs (np.ndarray): The location in the extracted box corresponding to each pixel.
            Shape=(npx,dim_ext). E.g. [z,sqrt(y**2+x**2)].
    
    Returns:
        ext_shape (np.ndarray): The shape of extracted box.
        ext_locs (np.ndarray): Pixel locations in the extracted box.
        mat_coo2loc (sp.sparse.csr_array): Matrix converting values of sampling pixels to extracted pixels.
    """
    # setup: int, nonnegativity
    box_coos = np.asarray(box_coos, dtype=int)
    box_locs = np.asarray(box_locs, dtype=int)
    box_locs -= np.min(box_locs, axis=0)

    # locations in the extracted box
    ext_locs = np.unique(box_locs, axis=0)
    ext_shape = np.max(ext_locs, axis=0) + 1
    loc_dict = {
        tuple(loc_i): i
        for i, loc_i in enumerate(ext_locs)
    }

    # construct conversion matrix
    # count degeneracy
    loc_count = np.zeros(len(ext_locs))
    icoo_arr = []
    iloc_arr = []
    for icoo, box_loc_i in enumerate(box_locs):
        iloc = loc_dict[tuple(box_loc_i)]
        loc_count[iloc] += 1
        icoo_arr.append(icoo)
        iloc_arr.append(iloc)
    # weight matrix values by 1/count
    loc_weight = (1/loc_count)[iloc_arr]
    mat_coo2loc = sp.sparse.csr_array(
        (loc_weight, (iloc_arr, icoo_arr)),
        shape=(len(loc_dict), len(icoo_arr))
    )

    # outputs
    return ext_shape, ext_locs, mat_coo2loc


#=========================
# box extraction: scalar
#=========================

def extract_box_avg(I, zyx, nzyx, box_rn, box_rt):
    """ Calculate the average pixel values in the sampling box of each point.

    Parallel processing: assumed npts >> the number of sampling box pixels,
    so that iterating over sampling pixels is more efficient.

    Args:
        I (np.ndarray): Tomo, shape=(nz,ny,nx).
        zyx (np.ndarray): Position of each point, shape=(npts,3), each item=[zi,yi,xi].
        nzyx (np.ndarray): Normal of each point, shape=(npts,3), each item=[nzi,nyi,nxi].
        box_rn (2-tuple of float): (min,max) extensions in normal direction.
        box_rt (float): max extension in tangent direction.

    Returns:
        values (np.ndarray): Value for each point, shape=(npts,).
    """
    # setup: sampling boxes, numbers
    tyx, tz = pcdutil.normals_tangent(nzyx)
    box_coos = gen_box_coords(box_rn, box_rt)
    npts = len(zyx)
    ncoos = len(box_coos)

    # image interpolation
    grids = [np.arange(I.shape[i]) for i in range(3)]
    I_interp = sp.interpolate.RegularGridInterpolator(grids, I)

    # values for each membrane point
    values = np.zeros(npts)
    # split box_coos into chunks: use all available threads
    nthreads = os.cpu_count()
    box_coos_chunk = np.array_split(np.arange(len(box_coos)), nthreads)
    # pool
    lock = multiprocessing.dummy.Lock()
    pool = multiprocessing.dummy.Pool()
    # sum values for i'th chunk of box_coos
    def calc_one(i):
        values_i = np.zeros(npts)
        for j in box_coos_chunk[i]:
            bc_j = box_coos[j]
            zyx_j = zyx + bc_j[0]*nzyx + bc_j[1]*tyx + bc_j[2]*tz
            values_i[:] += I_interp(zyx_j)
        # lock when writing to the shared array
        lock.acquire()
        values[:] += values_i
        lock.release()
    # parallel processing over sampling pixels
    pool.map(calc_one, range(nthreads))
    pool.close()

    # average
    values = values / ncoos
    return values


#=========================
# box extraction: tensor
#=========================

def extract_box_tensor(I, zyx, nzyx, box_coos, box_locs, normalize):
    """ Extract box for each point. Can be adapted to different boxes 

    Args:
        I (np.ndarray): Tomo, shape=(nz,ny,nx).
        zyx (np.ndarray): Position of each point, shape=(npts,3), each item=[zi,yi,xi].
        nzyx (np.ndarray): Normal of each point, shape=(npts,3), each item=[nzi,nyi,nxi].
        box_coos (np.ndarray): The coordinate of each pixel in the sampling box, shape=(npx,dim_box).
        box_locs (np.ndarray): The location in the extracted box corresponding to each pixel, shape=(npx,dim_ext).
        normalize (bool): Whether to z-score each box.

    Returns:
        ext_boxes (np.ndarray): Extracted box for each point, shape=(npts,*ext_shape).
            ext_shape is the shape of each extracted box.
    """
    # setup: sampling boxes, numbers
    tyx, tz = pcdutil.normals_tangent(nzyx)
    ext_shape, ext_locs, mat_coo2loc = gen_box_mapping(box_coos, box_locs)
    npts = len(zyx)

    # image interpolation
    grids = [np.arange(I.shape[i]) for i in range(3)]
    I_interp = sp.interpolate.RegularGridInterpolator(grids, I)

    # extracted box for each point
    ext_boxes = np.zeros((npts, *ext_shape))

    def calc_one(i):
        # points at box_coos
        zyx_coos = (
            zyx[i] + box_coos[:, :1]*nzyx[i]
            + box_coos[:, 1:2]*tyx[i] + box_coos[:, 2:]*tz[i]
        )
        # values at box_coos
        v_coos = I_interp(zyx_coos)
        # convert to extracted box
        ext_boxes[i][tuple(ext_locs.T)] = mat_coo2loc @ v_coos
        # normalize
        if normalize:
            box_mean = np.mean(ext_boxes[i], keepdims=True)
            box_std = np.std(ext_boxes[i], keepdims=True)
            ext_boxes[i] = (ext_boxes[i] - box_mean) / box_std

    # parallel processing over sampling pixels
    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(npts))
    pool.close()

    return ext_boxes

def extract_box_3d(I, zyx, nzyx, box_rn, box_rt, normalize=True):
    """ Extract 3d box for each point.

    Args:
        I (np.ndarray): Tomo, shape=(nz,ny,nx).
        zyx (np.ndarray): Position of each point, shape=(npts,3), each item=[zi,yi,xi].
        nzyx (np.ndarray): Normal of each point, shape=(npts,3), each item=[nzi,nyi,nxi].
        box_rn (2-tuple of float): (min,max) extensions in normal direction.
        box_rt (float): Max extension in tangent direction.
        normalize (bool): Whether to z-score each box.

    Returns:
        ext_boxes (np.ndarray): Extracted box for each point, shape=(npts,*ext_shape).
            ext_shape is the shape of each extracted box, [nz,ny,nx].
    """
    # setup box
    box_coos = gen_box_coords(box_rn, box_rt)
    box_locs = box_coos - box_coos.min(axis=0)
    # extract
    ext_boxes = extract_box_tensor(
        I, zyx, nzyx,
        box_coos=box_coos, box_locs=box_locs,
        normalize=normalize
    )

    return ext_boxes

def extract_box_2drot(I, zyx, nzyx, box_rn, box_rt, normalize=True):
    """ Extract 2d box for each point. Rotational averaged along normal axis.

    Args:
        I (np.ndarray): Tomo, shape=(nz,ny,nx).
        zyx (np.ndarray): Position of each point, shape=(npts,3), each item=[zi,yi,xi].
        nzyx (np.ndarray): Normal of each point, shape=(npts,3), each item=[nzi,nyi,nxi].
        box_rn (2-tuple of float): (min,max) extensions in normal direction.
        box_rt (float): max extension in tangent direction.
        normalize (bool): Whether to z-score each box.

    Returns:
        ext_boxes (np.ndarray): Extracted box for each point, shape=(npts,*ext_shape).
            ext_shape is the shape of each extracted box, [ny,nx].
    """
    # setup box
    box_coos = gen_box_coords(box_rn, box_rt)
    box_locs = np.stack([
        box_coos[:, 0]-box_coos[:, 0].min(),
        np.ceil(np.sqrt(box_coos[:, 1]**2+box_coos[:, 2]**2))
    ], axis=1
    ).astype(int)
    # extract
    ext_boxes = extract_box_tensor(
        I, zyx, nzyx,
        box_coos=box_coos, box_locs=box_locs,
        normalize=normalize
    )
    return ext_boxes


#=========================
# local max
#=========================

def localmax_neighbors(zyx, values, r_thresh):
    """ Find points with values larger than all its neighbors.

    Args:
        zyx (np.ndarray): Position of points, shape=(npts,dim).
        values (np.ndarray): Value for each point, shape=(npts,).
        r_thresh (float): Distance threshold for neighbors in graph construction.

    Returns:
        mask (np.ndarray): Mask of bool, shape=(npts,).
            zyx[mask] gives the selected points.
    """
    # construct graph
    g = pcdutil.neighbors_graph(zyx, r_thresh=r_thresh)

    # find if each point is local max
    npts = len(zyx)
    mask = np.zeros(npts, dtype=bool)

    def calc_one(i):
        mask[i] = values[i] > np.max(values[g.neighbors(i)])

    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(npts))
    pool.close()

    return mask

def localmax_perslice(zyx, values, r_thresh):
    """ Find points with values larger than all its neighbors in the same xy-slice.

    Args:
        zyx (np.ndarray): Position of points, shape=(npts,dim).
        values (np.ndarray): Value for each point, shape=(npts,).
        r_thresh (float): Distance threshold for neighbors in graph construction.

    Returns:
        mask (np.ndarray): Mask of bool, shape=(npts,).
            zyx[mask] gives the selected points.
    """
    mask = np.zeros(len(zyx), dtype=bool)
    # local max in each slice
    for z in np.unique(zyx[:, 0]):
        mask_z = zyx[:, 0] == z
        mask_z_nms = localmax_neighbors(
            zyx[mask_z], values[mask_z],
            r_thresh=r_thresh
        )
        mask[mask_z] = mask_z_nms
    return mask

def localmax_exclusion(zyx, values, r_thresh):
    """ Find the point with the largest value, then exclude this point and its neighbors, iteratively find the next largest one.

    Args:
        zyx (np.ndarray): Position of points, shape=(npts,dim).
        values (np.ndarray): Value for each point, shape=(npts,).
        r_thresh (float): Distance threshold for neighbors in graph construction.

    Returns:
        mask (np.ndarray): Mask of bool, shape=(npts,).
            zyx[mask] gives the selected points.
    """
    # construct graph
    g = pcdutil.neighbors_graph(zyx, r_thresh=r_thresh)
    g.vs["id"] = np.arange(g.vcount())
    g.vs["value"] = values

    # iteratively find the largest, and delete graph vertices
    id_arr = []
    while g.vcount() > 0:
        # current index of the max point
        imax = np.argmax(g.vs["value"])
        # save the original index of the max point
        id_arr.append(g.vs[imax]["id"])
        # update graph: delete max point and neighbors
        g.delete_vertices([imax, *g.neighbors(imax)])
    id_arr = np.asarray(id_arr)

    # make mask
    mask = np.zeros(len(zyx), dtype=bool)
    mask[id_arr] = True
    return mask


#=========================
# classification
#=========================

def classify_2drot_init(box_rn, box_rt):
    """ Initial means for classification on 2drot boxes.

    Args:
        box_rn (2-tuple of float): (min,max) extensions in normal direction.
        box_rt (float): max extension in tangent direction.
    
    Returns:
        means_init (np.ndarray): Initial mean for clusters. Shape=(2,ny,nx).
            means_init[0]: A particle-shaped image.
            means_init[1]: A membrane-shaped image.
    """
    # setup boxes the same way as in extract_box_2drot()
    # especially the mapping from coordinates to pixels
    box_coos = gen_box_coords(box_rn, box_rt)
    box_locs = np.stack([
        box_coos[:, 0]-box_coos[:, 0].min(),
        np.ceil(np.sqrt(box_coos[:, 1]**2+box_coos[:, 2]**2))
    ], axis=1
    ).astype(int)
    ext_shape, ext_locs, mat_coo2loc = gen_box_mapping(box_coos, box_locs)

    # assign values at object's coordinates to 1
    # membrane: horizontal plate
    values_mem = (box_coos[:, 0]**2<=1).astype(int)
    # particle: vertical stick + membrane
    values_part = values_mem + (box_coos[:, 1]**2+box_coos[:, 2]**2 <=1).astype(int)

    # generate images: assign pixels corresponding to objects to 1
    # use mat@values to binarize the result
    means_init = np.zeros((2, *ext_shape), dtype=float)
    means_init[0][tuple(ext_locs.T)] = (mat_coo2loc @ values_part)>0
    means_init[1][tuple(ext_locs.T)] = (mat_coo2loc @ values_mem)>0
    return means_init

def classify_2drot_boxes(boxes, box_rn, box_rt):
    """ GMM classification of 2drot boxes into particle-like and membrane-like clusters.

    Args:
        boxes (np.ndarray): Results of extract_box_2drot(). Shape=(nboxes,ny,nx).
        box_rn (tuple), box_rt (float): Ranges of the box. See doc of extract_box_2drot().
            Values should be the same as ones used in extract_box_2drot().
    
    Returns:
        memberships (np.ndarray): Membership of each box. Shape=(nboxes,).
            0 for particle-like, 1 for membrane-like.
    """
    # initialize mean
    means_init = classify_2drot_init(box_rn, box_rt)
    
    # preprocess data
    # reshape (n,ny,nx) to (n,ny*nx); z-score for each row
    def preprocess(arr):
        arr = arr.reshape((len(arr), -1))
        arr = (arr - np.mean(arr, axis=1))/np.std(arr, axis=1)
        return arr
    means_init = preprocess(means_init)
    boxes_flat = preprocess(boxes)

    # clustering
    clust = mixture.GaussianMixture(
        n_components=2,
        covariance_type="diag",
        means_init=means_init
    )
    clust.fit(boxes_flat)
    memberships = clust.predict(boxes_flat)
    return memberships
