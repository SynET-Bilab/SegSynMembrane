import numpy as np
from etsynseg import pcdutil

__all__ = [
    "refine_surface",
]

def refine_surface(
        zyx, sigma_normal, sigma_mesh, sigma_hull,
        target_spacing=1, B_bound=None
    ):
    """ Refine surface using mesh-based methods, mainly poisson reconstruction

    Args:
        zyx (np.ndarray): Points to be refined. Shape=(npts,dim), format [[z0,y0,x0],...].
        sigma_normal (float): Neighborhood radius for normal calculation
        sigma_mesh (float): Target spatial resolution for poisson reconstruction.
        sigma_hull (float): Length to extend in the normal direction when computing hull.
        target_spacing (float): Target spacing of points.
        B_bound (np.ndarray): The mask region with shape=(nz,ny,nx). Surface will be constrained inside.

    Returns:
        zyx_refine (np.ndarray): Points for the refined surface.
    """
    # create mesh
    pcd = pcdutil.normals_pointcloud(
        pcd=pcdutil.points2pointcloud(zyx),
        sigma=sigma_normal
    )
    mesh = pcdutil.reconstruct_mesh(
        pcd, target_width=sigma_mesh
    )

    # convex hull
    hull = pcdutil.convex_hull(
        pts=pcd.points,
        normals=pcd.normals,
        sigma=sigma_hull
    )

    # select center+surround region of the mesh
    mask_center = pcdutil.points_in_hull(mesh.vertices, hull)
    iv_center = np.nonzero(mask_center)[0]
    iv_surround = pcdutil.meshpoints_surround(mesh, iv_center)
    mesh = mesh.select_by_index(iv_surround)

    # subdivide to pixel scale
    mesh_div = pcdutil.subdivide_mesh(mesh, target_spacing=target_spacing)
    
    # remove duplicates
    zyx_div = pcdutil.points_deduplicate(np.asarray(mesh_div.vertices))

    # constrain points in the convex hull
    mask_inhull = pcdutil.points_in_hull(zyx_div, hull)
    zyx_refine = zyx_div[mask_inhull]

    # constrain in bound if provided
    if B_bound is not None:
        mask_inside = pcdutil.points_in_region(zyx_refine, B_bound)
        zyx_refine = zyx_refine[mask_inside]

    return zyx_refine
