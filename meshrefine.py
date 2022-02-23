""" refinement of segmented surface
note the order:
    refine_surface: zyx, the same as most other submodules
    normal_points and others: xyz, to interface with open3d
"""

import numpy as np
from scipy import spatial
import open3d as o3d
from etsynseg import utils

__all__ = [
    # common external uses
    "normals_points", "refine_surface",
    # specific external uses
    "create_pointcloud", "create_mesh_poisson",
    "convex_hull", "points_in_hull", "mesh_subdivide",
]

#=========================
# create mesh
#=========================

def create_pointcloud(xyz):
    """ create open3d point cloud from points (xyz)
    :param xyz: [[x1,y1,z1],...]
    :return: pcd
        pcd: open3d pointcloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def normals_pointcloud(pcd, sigma, pt_ref=None):
    """ estimate normals for point cloud
    :param pcd: open3d pointcloud
    :param sigma: neighborhood radius for normal calculation
    :param pt_ref: reference point inside, for orienting normals
    :return: pcd
        pcd: point cloud with normals
    """
    # calculate normals
    search_param = o3d.geometry.KDTreeSearchParamRadius(sigma)
    pcd.estimate_normals(search_param, fast_normal_computation=False)
    pcd.orient_normals_consistent_tangent_plane(10)

    # align normals using the reference point
    if pt_ref is not None:
        # direction from pt_ref to the closest point in pcd
        imin = np.argmin(np.linalg.norm(
            np.asarray(pcd.points)-pt_ref, axis=1))
        # align normals with this direction
        sign = np.sign(np.dot(pcd.normals[imin], pcd.points[imin]))
        if sign < 0:
            pcd.normals = o3d.utility.Vector3dVector(
                -np.asarray(pcd.normals))
    return pcd

def normals_points(xyz, sigma, pt_ref=None):
    """ estimate normals for points
    :param xyz: [[x1,y1,z1],...]
    :param sigma: neighborhood radius for normal calculation
    :param pt_ref: reference point inside, for orienting normals
    :return: normals
        normals: [[nx1,ny1,nz1],...]
    """
    pcd = create_pointcloud(xyz)
    pcd = normals_pointcloud(pcd, sigma, pt_ref)
    normals = np.asarray(pcd.normals)
    return normals

def create_mesh_poisson(pcd, resolution):
    """ create mesh using poisson surface reconstruction
    :param pcd: open3d pointcloud, has normals
    :param resolution: resolution in voxel
    :return: mesh
        mesh: open3d mesh
    """
    # set to int, required by open3d<0.15
    width = max(1, int(resolution))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, width=width)
    return mesh


#=========================
# refine mesh
#=========================

def convex_hull(pts, normals=None, factor_extend=1):
    """ compute convex hull
    :param pts: points
    :param normals, factor_extend: extend region by shifting points along normals by the factor
    :return: hull
        hull: scipy.spatial.ConvexHull object
    """
    pts = np.asarray(pts)
    if (factor_extend > 0) and (normals is not None):
        fnormals = factor_extend * np.asarray(normals)
        pts_ext = np.concatenate([pts, pts+fnormals, pts-fnormals])
    else:
        pts_ext = pts
    hull = spatial.ConvexHull(pts_ext)
    return hull

def points_in_hull(pts, hull):
    """ find which points are in the hull
    :param pts: points
    :param hull: scipy.spatial.ConvexHull object
    :return: index
        index: index of points in the hull
    """
    # https://stackoverflow.com/questions/31404658/check-if-points-lies-inside-a-convex-hull
    eps = np.finfo(np.float64).eps
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1:]
    mask = np.all(np.asarray(pts) @ A.T + b.T < eps, axis=1)
    index = np.nonzero(mask)[0]
    return index

def points_surround(mesh, iv_center):
    """ find points surrounding the center ones
    :param mesh: open3d mesh
    :param iv_center: index of points in the center
    :return: iv_surround
        iv_surround: index of center+surround points
    """
    # mesh for center
    # no cleanup: so indexes before and after can match
    mcenter = mesh.select_by_index(iv_center, cleanup=False)
    # vertex indexes (in mcenter) on boundary
    # allow_boundary_edges: false so as to find mostly edges
    iv_bound = np.unique(
        mcenter.get_non_manifold_edges(allow_boundary_edges=False))

    # find in the original mesh
    tri = np.asarray(mesh.triangles)
    # inside nonzero: find triangles if any of its vertices belongs to the boundary
    # iv_center[iv_bound]: convert indexing from mcenter to mesh
    itri_bound = np.nonzero(
        np.sum(np.isin(tri, iv_center[iv_bound]), axis=1)
    )[0]
    # index (in mesh) for center + surround
    iv_surround = np.unique(np.concatenate(
        [iv_center, tri[itri_bound].ravel()]))
    return iv_surround

def length_edges(mesh):
    """ calculate the length of edges in the mesh
    :param mesh: open3d mesh
    :return: lengths
        lengths: array of lengths for each edge
    """
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    vs = np.asarray(lineset.points)
    es = np.asarray(lineset.lines)
    lengths = np.max(np.abs(vs[es[:, 0]] - vs[es[:, 1]]), axis=1)
    return lengths

def mesh_subdivide(mesh, target_len=1):
    """ subdivide-simplify mesh till the max length of edges <= target_len
    :param mesh: open3d mesh
    :param target_len: target edge length
    :return: mdiv
        mdiv: subdivided open3d mesh
    """
    # copy mesh
    mdiv = o3d.geometry.TriangleMesh()
    mdiv.vertices = mesh.vertices
    mdiv.triangles = mesh.triangles

    # cycle: subdivide - simplify
    # cleanup: removed non-manifold edges, but unreferenced vertices are kept
    while np.max(length_edges(mdiv)) > target_len:
        mdiv = mdiv.subdivide_loop()
        mdiv = mdiv.simplify_vertex_clustering(
            voxel_size=0.5,
            contraction=o3d.geometry.SimplificationContraction.Quadric
        )
        mdiv = mdiv.remove_non_manifold_edges()
    return mdiv


#=========================
# workflow
#=========================

def refine_surface(zyx, sigma_normals, sigma_mesh):
    """ refine surface using mesh-based methods, mainly poisson reconstruction
    :param zyx: points
    :param sigma_normals: length scale for normal estimation
    :param sigma_mesh: spatial resolution for poisson reconstruction
    :return: zyx_refine
        zyx_refine: points for the refined surface
    """
    # create mesh
    xyz = utils.reverse_coord(zyx)
    pcd = normals_pointcloud(
        pcd=create_pointcloud(xyz=xyz),
        sigma=sigma_normals
    )
    mesh = create_mesh_poisson(pcd, resolution=sigma_mesh)

    # convex hull
    hull = convex_hull(pts=pcd.points, normals=pcd.normals)

    # select center+surround region of the mesh
    iv_center = points_in_hull(mesh.vertices, hull)
    iv_surround = points_surround(mesh, iv_center)
    mesh = mesh.select_by_index(iv_surround)

    # subdivide to pixel scale
    mdiv = mesh_subdivide(mesh, target_len=1)
    mdiv = mdiv.select_by_index(
        points_in_hull(mdiv.vertices, hull)
    )

    # convert to image
    zyx_raw = utils.reverse_coord(np.asarray(mdiv.vertices))
    shape = np.ceil(np.max(zyx, axis=0)).astype(np.int_)
    B_refine = utils.coord_to_mask(zyx_raw, shape)
    B_refine = next(utils.extract_connected(B_refine, connectivity=3))[1]
    zyx_refine = utils.mask_to_coord(B_refine)
    return zyx_refine
