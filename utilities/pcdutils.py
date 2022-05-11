""" Utilities for dealing with pointcloud-like data.
"""
import numpy as np
import scipy as sp
import open3d as o3d

__all__ = [
    # conversion
    "pixels2points", "points2pixels", "reverse_coord", "points2pointcloud",
    # normals
    "normals_pointcloud", "normals_points",
    # convex hull
    "convex_hull", "points_in_hull",
    # mesh
    "reconstruct_mesh", "subdivide_mesh", "meshpoints_surround",
    # grids
    "wireframe_length"
]

#=========================
# conversion
#=========================

def pixels2points(I):
    """ Convert an image to points corresponding to nonzero pixels.
    
    Args:
        I (np.ndarray): Image with shape = (ny,nx) for 2d or (nz,ny,nx) for 3d.
    
    Returns:
        pts (np.ndarray): Array of points with shape=(npts,I.ndim). Format of each point is [y,x] for 2d or [z,y,x] for 3d.
    """
    pts = np.argwhere(I)
    return pts

def points2pixels(pts, shape):
    """ Convert points to an image with 1's on point locations (rounded) and 0's otherwise.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,I.ndim). Format of each point is [y,x] for 2d or [z,y,x] for 3d.
        shape (tuple or list): Shape of the target image, (ny,nx) for 2d or (nz,ny,nx) for 3d.
    
    Returns:
        I (np.ndarray): Image with shape = (ny,nx) for 2d or (nz,ny,nx) for 3d.
    """
    # round to int
    pts = np.round(np.asarray(pts)).astype(int)

    # remove out-of-bound points
    dim = pts.shape[1]
    mask = np.ones(len(pts), dtype=bool)
    for i in range(dim):
        mask_i = (pts[:, i]>=0) & (pts[:, i]<=shape[i]-1)
        mask = mask & mask_i
    pts = pts[mask]

    # assign to pixels
    I = np.zeros(shape, dtype=int)
    index = tuple(pts.T)
    I[index] = 1
    return I

def reverse_coord(pts):
    """ Reverse the coordinate of points with shape=(npts,dim) along axis-1, e.g. [[zi,yi,xi],...] -> [[xi,yi,zi],...]
    
    Args:
        pts (np.ndarray): Points.
    
    Returns:
        pts_rev (np.ndarray): Points with reversed coordinates.
    """
    pts = np.asarray(pts)
    index_rev = np.arange(pts.shape[1])[::-1]
    return pts[:, index_rev]

def points2pointcloud(pts, normals=None):
    """ Convert points to open3d pointcloud.

    Args:
        pts (np.ndarray): Points with shape=(npts,dim).
        normals (np.ndarray, optional): Normals of the points, with shape=(npts,dim).

    Returns:
        pcd (open3d.geometry.PointCloud): Open3d pointcloud 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


#=========================
# normals
#=========================

def normals_pointcloud(pcd, sigma, pt_ref=None):
    """ Estimate normals for point cloud.

    Normals are oriented consistently. If a reference point (pt_ref) is provided, then normals point from pt_ref to pcd.

    Args:
        pcd (open3d.geometry.PointCloud): open3d pointcloud
        sigma (float): Neighborhood radius for normal estimation.
        pt_ref (np.ndarray, optional): Reference point 'inside'. Formated to be comparable with pcd.points[0].
    Returns:
        pcd (open3d.geometry.PointCloud): Point cloud with normals.
    """
    # calculate normals
    search_param = o3d.geometry.KDTreeSearchParamRadius(sigma)
    pcd.estimate_normals(
        search_param, fast_normal_computation=False
    )

    # orient: used 10 nearest neighbors for construction
    pcd.orient_normals_consistent_tangent_plane(10)

    # align normals using the reference point
    if pt_ref is not None:
        # direction from pt_ref to the closest point in pcd
        imin = np.argmin(np.linalg.norm(
            np.asarray(pcd.points)-pt_ref, axis=1
        ))
        # align normals with this direction
        sign = np.sign(
            np.dot(pcd.normals[imin], pcd.points[imin]-pt_ref)
        )
        if sign < 0:
            pcd.normals = o3d.utility.Vector3dVector(
                -np.asarray(pcd.normals)
            )
    return pcd

def normals_points(pts, sigma, pt_ref=None):
    """ Estimate normals for points.
    
    Wraps pcdutils.normals_pointcloud.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).
        sigma (float): Neighborhood radius for normal calculation
        pt_ref (np.ndarray, optional): Reference point 'inside'.
    
    Return:
        normals (np.ndarray): Array of normals for each point, with shape=(npts,dim).
    """
    pcd = points2pointcloud(pts)
    pcd = normals_pointcloud(pcd, sigma, pt_ref)
    normals = np.asarray(pcd.normals)
    return normals


#=========================
# convex hull
#=========================

def convex_hull(pts, normals=None, sigma=1):
    """ Compute the convex hull of points.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).
        normals (np.ndarray): Array of normals for each point, with the same shape as pts.
        sigma (float): Shift points along normals by + and - sigma before calculating the hull, to allow for some margin.
    
    Returns:
        hull (scipy.spatial.ConvexHull): The convex hull.
    """
    pts = np.asarray(pts)
    if (sigma > 0) and (normals is not None):
        normals = np.asarray(normals)
        fnormals = sigma * normals
        pts_ext = np.concatenate([pts, pts+fnormals, pts-fnormals])
    else:
        pts_ext = pts
    hull = sp.spatial.ConvexHull(pts_ext)
    return hull

def points_in_hull(pts, hull):
    """ Find points in the convex hull.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).
        hull (scipy.spatial.ConvexHull): The convex hull.

    Returns:
        index (np.ndarray): Array indexes of points in the hull.

    References:
        method: https://stackoverflow.com/questions/31404658/check-if-points-lies-inside-a-convex-hull
    """
    eps = np.finfo(np.float64).eps
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1:]
    mask = np.all(np.asarray(pts) @ A.T + b.T < eps, axis=1)
    index = np.nonzero(mask)[0]
    return index


#=========================
# mesh
#=========================

def reconstruct_mesh(pcd, target_width):
    """ Poisson surface reconstruction.

    Args:
        pcd (open3d.geometry.pointcloud): Pointcloud with normals.
        target_width (float): The target width of the finest level octree cells.
    Returns: mesh
        mesh (open3d.geometry.TriangleMesh): The reconstructed mesh.
    """
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, width=target_width
    )
    return mesh

def subdivide_mesh(mesh, target_spacing=1):
    """ Subdivide mesh till the max distance between nearest neighbors <= target_spacing

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh.
        target_spacing (float): Target spacing between points.
    
    Returns:
        mesh_div (open3d.geometry.TriangleMesh): Subdivided mesh.
    
    Notes:
        Sometimes may encounter "non-manifold edges" warning, which does not seem to heavily affect the result.
    """
    # copy mesh
    mesh_div = o3d.geometry.TriangleMesh()
    mesh_div.vertices = mesh.vertices
    mesh_div.triangles = mesh.triangles

    # max nearest neighbor distance
    pts = np.asarray(mesh_div.vertices)
    kdtree = sp.spatial.KDTree(pts)
    dists = kdtree.query(pts, k=2, p=1)[0][:, 1]
    max_dist = np.max(dists)

    # niter-round subdivision
    # niter: determined according to max 1nn distance
    niter = int(np.ceil(np.log2(max_dist/target_spacing)))
    if niter >= 1:
        mesh_div = mesh_div.subdivide_loop(number_of_iterations=niter)

    return mesh_div

def meshpoints_surround(mesh, idx_center):
    """ Find points in a mesh which surround the center ones.

    Args:
        mesh (open3d.geometry.TriangleMesh): Mesh.
        idx_center (np.ndarray): Index of points in the center.
    
    Returns:
        idx_surround (np.ndarray): Index of center+surround points.
    """
    # mesh for center
    # no cleanup: so indexes before and after can match
    mcenter = mesh.select_by_index(idx_center, cleanup=False)

    # vertex indexes (in mcenter) on boundary
    # allow_boundary_edges: false so as to find mostly edges
    idx_bound = np.unique(
        mcenter.get_non_manifold_edges(allow_boundary_edges=False)
    )

    # find in the original mesh
    # inside nonzero: find triangles if any of its vertices belongs to the boundary
    # idx_center[idx_bound]: convert indexing from mcenter to mesh
    tri = np.asarray(mesh.triangles)
    itri_bound = np.nonzero(
        np.sum(np.isin(tri, idx_center[idx_bound]), axis=1)
    )[0]

    # index (in mesh) for center + surround
    idx_surround = np.unique(np.concatenate(
        [idx_center, tri[itri_bound].ravel()]
    ))
    return idx_surround


#=========================
# grid tools
#=========================

def wireframe_length(pts_net, axis=0):
    """ Calculate total lengths of wireframe along one axis.

    Input can be either net-shaped points or flattened points.

    Args:
        pts_net (np.ndarray): Net-shaped nu*nv points arranged in net-shape, with shape=(nu,nv,dim). Or flattened points with shape=(nu,dim).
        axis (int): 0 for u-direction, 1 for v-direction.

    Returns:
        wires (np.ndarray): [len0,len1,...]. nv elements if axis=u, nu elements if axis=v.
    """
    # A, B - axes
    # [dz,dy,dx] along A for each B
    diff_zyx = np.diff(pts_net, axis=axis)
    # len of wire segments along A for each B
    segments = np.linalg.norm(diff_zyx, axis=-1)
    # len of wire along A for each B
    wires = np.sum(segments, axis=axis)
    return wires

def points_deduplicate(pts, round2int=True):
    """ Deduplicate points.

    Remove points that duplicate previous ones. Optionally round the points to integers first.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).
        round2int (bool): If round to integer first.

    Returns:
        pts_dedup (np.ndarray): Deduplicated array of points, with shape=(npts_dedup,dim).
    """
    if round2int:
        pts = np.round(pts).astype(int)
    diff = np.abs(np.diff(pts, axis=0)).max(axis=1)
    pts_dedup = np.concatenate([pts[:1], pts[1:][diff > 0]])
    return pts_dedup
