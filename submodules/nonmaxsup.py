""" Non-maximum suppression (NMS).

NMS using the Gonzalez-Woods method (8-neighbors, no interpolation).

"""

from functools import reduce
import multiprocessing.dummy
import numpy as np

__all__ = [
    "nms2d", "nms3d",
]

def gen_mask_boundary(I):
    """ Generate mask where pixels on the boundary are 0.

    Args:
        I (np.ndarray): Image in 2d.
    
    Returns:
        mask (np.ndarray): The mask.
    """
    mask = np.ones(I.shape, dtype=bool)
    for i in [0, -1]:
        mask[i, :] = 0  # y=0,ny
        mask[:, i] = 0  # x=0,nx
    return mask

def slice_adjacent(x, y):
    """ Generate slicing for adjacent pixels.

    Example:
        For image I and mask pts, if x="+", y="-", then I[sub_I][pts[sub_pts]] are points on the right(x+)-below(y-).
    
    Args:
        x, y (str): One of "+","-","0".

    Returns:
        sub_I, sub_pts (2-tuple of slice for each): See example for their meanings.
    """
    s = {
        "0": {
            "pts": slice(None, None),
            "I": slice(None, None)
        },
        "-": {
            "pts": slice(1, None),
            "I": slice(None, -1)
        },
        "+": {
            "I": slice(1, None),
            "pts": slice(None, -1)
        }
    }
    sub_I = (s[y]["I"], s[x]["I"])
    sub_pts = (s[y]["pts"], s[x]["pts"])
    return sub_I, sub_pts

def compare_local(N_range, x, y, I, N, mask):
    """ Compare local pixels.

    N_range, x, y indicate the direction to consider. E.g. N_range=(0, np.pi/8), x="+", y="0".

    Args:
        N_range (2-tuple): The range of angles in the normal direction, (N_min,N_max).
        x, y (str): Relative position of adjacent pixel, "+"/"-"/"0".
        I (np.ndarray): 2d image, with shape=(ny,nx).
        N (np.ndarray): Angles of the normal direction. Values are within [0,pi), and shape=I.shape.
        mask (np.ndarray): Additional mask with dtype=bool and with shape=I.shape.
    
    Returns:
        local_max (np.ndarray): bool, shape=I.shape, true if the pixel is local max.
        local_supp (np.ndarray): bool, shape=I.shape, true if the pixel is suppressed.
    """
    # pts in N_range
    pts = mask & (N >= N_range[0]) & (N < N_range[1])

    # find pixels that are local max or suppressed
    # max_plus/minus: true on pixels > its adjacent
    # supp_plus/minus: true on pixels being suppressed
    def init():
        return np.zeros(I.shape, dtype=bool)

    # plus direction
    sub_I, sub_pts = slice_adjacent(x=x, y=y)
    max_plus = init()
    max_plus[pts] = I[pts] > I[sub_I][pts[sub_pts]]
    supp_plus = init()
    supp_plus[sub_I][pts[sub_pts]] = max_plus[pts]

    # minus direction
    invert = {"0": "0", "+": "-", "-": "+"}
    sub_I, sub_pts = slice_adjacent(x=invert[x], y=invert[y])
    max_minus = init()
    max_minus[pts] = I[pts] > I[sub_I][pts[sub_pts]]
    supp_minus = init()
    supp_minus[sub_I][pts[sub_pts]] = max_minus[pts]

    # combine both directions
    # max: in both; supp: in either
    local_max = np.logical_and(max_plus, max_minus)
    local_supp = np.logical_or(supp_plus, supp_minus)

    return local_max, local_supp

def nms2d(I, O, suppress=True):
    """ Non-maximum suppresion of 2d image.

    Args:
        I (np.ndarray): Image with shape=(ny,nx).
        O (np.ndarray): Orientation of ridgelike features, with the same shape as I.
        suppress (bool): If true, then the suppressed pixels are excluded.

    Returns:
        B (np.ndarray): A binary image with 1's on pixels that are local max.
    """
    # global masks
    mask = gen_mask_boundary(I)

    # normal direction, mod pi
    N = np.mod(O+np.pi/2, np.pi)

    # local max in each direction
    directions = [
        dict(N_range=(0, np.pi/8), x="+", y="0"),
        dict(N_range=(np.pi/8, np.pi*3/8), x="+", y="+"),
        dict(N_range=(np.pi*3/8, np.pi*5/8), x="0", y="+"),
        dict(N_range=(np.pi*5/8, np.pi*7/8), x="-", y="+"),
        dict(N_range=(np.pi*7/8, np.pi), x="-", y="0")
    ]
    max_arr = []
    supp_arr = []
    kwargs = dict(I=I, N=N, mask=mask)
    for d in directions:
        max_d, supp_d = compare_local(d["N_range"], d["x"], d["y"], **kwargs)
        max_arr.append(max_d)
        supp_arr.append(supp_d)

    # combine all directions
    local_max = reduce(np.logical_or, max_arr)
    local_supp = reduce(np.logical_or, supp_arr)

    # exclude suppressed pixels
    if suppress:
        local_max = np.logical_and(
            local_max, np.logical_not(local_supp)
        )
    
    # return int-type array
    B = local_max.astype(int)
    return B

def nms3d(I, O, suppress=True):
    """ Non-maximum suppresion of stack of 2d images.

    Args:
        I (np.ndarray): Image with shape=(nz,ny,nx).
        O (np.ndarray): Orientation of ridgelike features for each slice, with the same shape as I.
        suppress (bool): If true, then the suppressed pixels are excluded.

    Returns:
        B (np.ndarray): A binary image with 1's on pixels that are local max.
    """
    # find local max for each slice
    B = np.zeros(I.shape, dtype=int)
    nz = I.shape[0]

    # parallel computing
    def calc_one(i):
        B[i] = nms2d(I[i], O[i], suppress=suppress)
        
    pool = multiprocessing.dummy.Pool()
    pool.map(calc_one, range(nz))
    pool.close()

    return B
