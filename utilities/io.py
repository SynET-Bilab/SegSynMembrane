""" input-output
"""

import tempfile
import subprocess
import numpy as np
import pandas as pd
import mrcfile

__all__ = [
    # tomo io
    "read_tomo", "write_tomo",
    # model io
    "read_model", "write_model", "write_points"
]


#=========================
# tomo
#=========================

def read_tomo(tomo_file, mode="mmap", xrange=(None, None), yrange=(None, None), zrange=(None, None)):
    """ Read tomo file.
    
    Tomo file is in mrc format.
    Can clip tomo within some range.
    Assumed pixel size in all dimensions are the same, so a single pixel size is returned.

    Args:
        tomo_file (str): Input filename for tomo.
        mode (str): Mode for opening the file, 'open' or 'mmap'.
        xrange, yrange, zrange (2-tuple): Clip tomo to this range. E.g. xrange=(xmin,xmax), tomo[:,:,xmin:xmax].

    Returns:
        I (np.ndarray or np.memmap): 3d tomo image.
        pixel_A (float): Pixel size in angstrom.
    """
    # setup mode
    if mode == "open":
        fopen = mrcfile.open
    elif mode == "mmap":
        fopen = mrcfile.mmap
    else:
        raise ValueError("Mode should be 'open' or 'mmap'.")

    # setup clipping range
    sub = (slice(*zrange), slice(*yrange), slice(*xrange))
    
    with fopen(tomo_file, permissive=True) as mrc:
        # read subset of tomo
        I = mrc.data[sub]
        # read pixel size (3-tuple from mrcfile)
        # convert to one value
        pixel_A = mrc.voxel_size
        pixel_A = np.mean(pixel_A.item())
    return I, pixel_A

def write_tomo(tomo_file, I, pixel_A=1, dtype=None):
    """ Write 3d tomo data to mrc file.

    Datatypes that are mrc-compatible:
    https://mrcfile.readthedocs.io/en/latest/usage_guide.html#data-types
    int8/16, uint8/16, float32, complex64

    Args:
        tomo_file (str): Output filename for tomo.
        I (np.ndarray or np.memmap): 3d tomo image.
        pixel_A (float): Pixel size in angstrom.
        dtype (type): Target data type. If None, then will use I.dtype.
    """
    # set datatype
    I = np.asarray(I)

    if dtype is None:
        dtype = I.dtype
    if np.issubdtype(dtype, np.signedinteger):
        if dtype not in [np.int8, np.int16]:
            dtype = np.int16
    elif np.issubdtype(dtype, np.unsignedinteger):
        if dtype not in [np.uint8, np.uint16]:
            dtype = np.uint16
    elif np.issubdtype(dtype, np.floating):
        if dtype not in [np.float32]:
            dtype = np.float32
    elif np.issubdtype(dtype, np.complexfloating):
        if dtype not in [np.complex64]:
            dtype = np.complex64

    I = I.astype(dtype)

    # write new mrc
    # note: first set data, then set voxel_size
    with mrcfile.new(tomo_file, overwrite=True) as mrc:
        mrc.set_data(I)
        mrc.voxel_size = pixel_A


#=========================
# imod model
#=========================

def read_point(point_file, dtype_z=int):
    """ Read imod point file.
    
    Point file is converted from imod model file by model2point -ob in.mod out.pt.
    The file should contain 5 columns, corresponding to [object,contour,x,y,z].
    {object,contour} are 1-based.
    {x,y,z} are 0-based.

    Args:
        point_file (str): Filename of point.
        dtype_z (type): Datatype for z. Default is int, for easier slicing.

    Returns:
        model (pd.DataFrame): Dataframe object for the points, with columns=[object,contour,x,y,z].
    """
    # load point
    point = np.loadtxt(point_file)
    if point.shape[1] != 5:
        raise ValueError("Point file should contain five columns, corresponding to object,contour,x,y,z.")

    # convert to dataframe
    cols = ["object", "contour", "x", "y", "z"]
    dtypes = [int, int, float, float, dtype_z]
    data = {
        cols[i]: pd.Series(point[:, i], dtype=dtypes[i])
        for i in range(5)
    }
    model = pd.DataFrame(data)
    return model

def read_model(model_file, dtype_z=int):
    """ Read imod model file.

    Fields extracted are [object,contour,x,y,z]. {object,contour} are 1-based.
    {x,y,z} are 0-based.

    Args:
        model_file (str): Filename of imod model.
        dtype_z (type): Datatype for z. Default is int, for easier slicing.

    Returns:
        model (pd.DataFrame): Dataframe object for the points, with columns=[object,contour,x,y,z].
    """
    with tempfile.NamedTemporaryFile(suffix=".pt") as temp_file:
        # model2point
        point_file = temp_file.name
        cmd = f"model2point -ob {model_file} {point_file} >/dev/null"
        subprocess.run(cmd, shell=True, check=True)

        # point to DataFrame
        model = read_point(point_file, dtype_z=dtype_z)
        
        return model

def write_model(model_file, model):
    """ Write imod model file.

    Model should contain columns [object,contour,x,y,z].
    {object,contour} are 1-based.
    {x,y,z} are 0-based.   

    Args:
        model_file (str): Filename of output imod model.
        model (np.ndarray or pd.DataFrame): Data for the model. Will be converted by np.asarray.
    """
    model = np.asarray(model)

    with tempfile.NamedTemporaryFile(suffix=".pt") as temp_file:
        # save as point file
        # format: int for object,contour; float for x,y,z
        point_file = temp_file.name
        np.savetxt(
            point_file, model,
            fmt=(['%d']*2 + ['%.2f']*3)
        )

        # point2model: -op for open contour
        cmd = f"point2model -op {point_file} {model_file} >/dev/null"
        subprocess.run(cmd, shell=True, check=True)

def write_points(zyx_arr, break_contour=None, save=None):
    """ Convert points to model DataFrame, and write.
    
    Each element in the point array is one object.
    Points at different z's are assigned to different contours.
    Points in the same xy plane are assigned to the same contour if break_contour=None,
    or are broken into different contours if their distance is > break_contour(float).

    Args:
        zyx_arr (list of np.ndarray): Array of points.
            zyx_arr = [zyx_0,zyx_1,...],
            zyx_i=[[z0,y0,x0],...], with shape=(npts_i,3).
        break_contour (float, optional): When to break points into different contours.
        save (str, optional): Filename to save the model.

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

    # save model
    if save is not None:
        write_model(save, model)
    return model
