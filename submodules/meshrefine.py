import numpy as np
from etsynseg import pcdutils

__all__ = [
    "refine_surface",
]

def refine_surface(
        zyx,
        sigma_normal, sigma_mesh, sigma_hull,
        target_spacing=1, mask_bound=None, return_mesh=False
    ):
    """ Refine surface using mesh-based methods, mainly poisson reconstruction

    Args:
        zyx (np.ndarray): Points with shape=(npts, ndim) and in format [[z0,y0,x0],...].
        sigma_normal (float): Neighborhood radius for normal calculation
        sigma_mesh (float): Target spatial resolution for poisson reconstruction.
        sigma_hull (float): Length to extend in the normal direction when computing hull.
        target_spacing (float): Target spacing of points.
        mask_bound (np.ndarray, optional): Mask for boundary, with shape=(nz,ny,nx).

    Returns:
        zyx_refine (np.ndarray): Points for the refined surface.
    """
    # create mesh
    pcd = pcdutils.normals_pointcloud(
        pcd=pcdutils.points2pointcloud(zyx),
        sigma=sigma_normal
    )
    mesh = pcdutils.reconstruct_mesh(
        pcd, target_width=sigma_mesh
    )

    # convex hull
    hull = pcdutils.convex_hull(
        pts=pcd.points,
        normals=pcd.normals,
        sigma=sigma_hull
    )

    # select center+surround region of the mesh
    iv_center = pcdutils.points_in_hull(mesh.vertices, hull)
    iv_surround = pcdutils.meshpoints_surround(mesh, iv_center)
    mesh = mesh.select_by_index(iv_surround)

    # subdivide to pixel scale
    mesh = pcdutils.subdivide_mesh(mesh, target_spacing=target_spacing)
    
    # remove duplicates
    pts = np.round(np.asarray(mesh.vertices)).astype(int)
    pts = np.unique(pts, axis=0)  # np.unique also sorts the points

    # constrain points in the convex hull
    idx_inhull = pcdutils.points_in_hull(pts, hull)
    zyx_refine = pts[idx_inhull]

    # constrain in mask_bound if provided
    if mask_bound is not None:
        shape = mask_bound.shape
        B_refine = mask_bound * pcdutils.points2pixels(zyx_refine, shape)
        zyx_refine = pcdutils.pixels2points(B_refine)

    return zyx_refine
