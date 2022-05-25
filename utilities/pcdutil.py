""" Utilities for dealing with pointcloud-like data.
"""
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import decomposition
import igraph
import open3d as o3d

__all__ = [
    # conversion
    "pixels2points", "points2pixels", "reverse_coord", "points2pointcloud",
    # misc
    "points_range", "points_deduplicate", "points_deduplicate_count", "points_distance",
    "points_in_region", "orients_absdiff", "wireframe_length",
    # normals
    "normals_gen_ref", "normals_pointcloud", "normals_points",
    # convex hull
    "convex_hull", "points_in_hull",
    # mesh
    "reconstruct_mesh", "subdivide_mesh", "meshpoints_surround",
    # graph
    "neighbors_graph", "graph_components", "neighboring_components",
    # sorting
    "sort_pcds_by_ref", "sort_pts_by_guide_2d", "sort_pts_by_guide_3d"
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

def points2pixels(pts, shape=None, dtype=int):
    """ Convert points to an image with 1's on point locations (rounded) and 0's otherwise.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,I.ndim). Format of each point is [y,x] for 2d or [z,y,x] for 3d.
        shape (tuple): Shape of the target image, (ny,nx) for 2d or (nz,ny,nx) for 3d.
        dtype (type): Datatype for the image, int or bool.
    
    Returns:
        I (np.ndarray): Image with shape = (ny,nx) for 2d or (nz,ny,nx) for 3d.
    """
    # round to int
    pts = np.round(np.asarray(pts)).astype(int)

    # setup shape
    if shape is None:
        shape = np.max(pts, axis=0) + 1

    # remove out-of-bound points
    dim = pts.shape[1]
    mask = np.ones(len(pts), dtype=bool)
    for i in range(dim):
        mask_i = (pts[:, i]>=0) & (pts[:, i]<=shape[i]-1)
        mask = mask & mask_i
    pts = pts[mask]

    # assign to pixels
    I = np.zeros(shape, dtype=dtype)
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
# misc
#=========================

def points_range(pts, margin=0):
    """ Calculate the range of points.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).
        margin (float or tuple): Margin in each direction to be added to the range.

    Returns:
        low (np.ndarray of int): Point at the lowest end, with shape=(dim,).
        high (np.ndarray of int): Point at the highest end, with shape=(dim,).
        shape (tuple of int): Shape of image that can contain the points, high-low+1.
    """
    margin = np.ceil(np.asarray(margin)).astype(int)
    low = np.floor(np.min(pts, axis=0)).astype(int) - margin
    high = np.ceil(np.max(pts, axis=0)).astype(int) + margin
    shape = tuple(high - low + 1)
    return low, high, shape

def points_deduplicate(pts):
    """ Deduplicate int points while retaining the order according to the first appearances.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).

    Returns:
        pts_dedup (np.ndarray): Deduplicated array of points, with shape=(npts_dedup,dim).
    """
    # round
    pts = np.round(pts).astype(int)
    # convert to tuple, deduplicate using dict
    pts = [tuple(pt) for pt in pts]
    pts_dedup = dict.fromkeys(pts).keys()
    # convert to np.ndarray
    pts_dedup = np.array(list(pts_dedup))
    return pts_dedup

def points_deduplicate_count(pts):
    """ Deduplicate points while retaining the order. Count the number of appearances.

    This is slower that points_deduplicate, so creating a separate function.
    
    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).

    Returns:
        pts_dedup (np.ndarray): Deduplicated array of points, with shape=(npts_dedup,dim).
        pts_count (np.ndarray): The number of appearance of points, with shape=(npts_dedup,dim).
    """
    # round
    pts = np.round(pts).astype(int)
    # count using dict
    pts_dict = {}
    for pt in pts:
        pt = tuple(pt)
        if pt in pts_dict:
            pts_dict[pt] += 1
        else:
            pts_dict[pt] = 1
    # convert to np.ndarray
    pts_dedup = np.asarray(list(pts_dict.keys()))
    pts_count = np.asarray(list(pts_dict.values()))
    return pts_dedup, pts_count

def points_distance(pts1, pts2, return_2to1=False):
    """ Calculate distances between two point arrays.

    Args:
        pts1, pts2 (np.ndarray): Two pointclouds, each with shape=(nptsi,dim), i=1,2.
        return_2to1 (bool): Whether to return the distance from pts2 to pts1.
            If True, returns dist1, dist2.
            If False, returns dist1.
    
    Returns:
        dist1 (np.ndarray): Distance of each point in pts1 to its nearest point in pts2, shape=(npts1,dim).
        dist2 (np.ndarray, optional): Likewise for pts2, shape=(npts2,dim).
    """
    kdtree2 = sp.spatial.KDTree(pts2)
    dist1, _ = kdtree2.query(pts1, workers=-1)
    if not return_2to1:
        return dist1
    else:
        kdtree1 = sp.spatial.KDTree(pts1)
        dist2, _ = kdtree1.query(pts2, workers=-1)
        return dist1, dist2

def points_in_region(zyx, region):
    """ Select points inside a mask region.

    Args:
        zyx (np.ndarray): Points with shape=(nzyx,dim) and order=[z,y,x].
        region (np.ndarray): The region represented as a binary image, with shape=(nz,ny,nx).

    Returns:
        mask (np.ndarray): Array of bools indicating whether each point is in the region, shape=(nzyx,).
            zyx[mask] gives points in the region.
    """
    # mask for points inside region.shape
    dim = zyx.shape[1]
    mask_shape = np.ones(len(zyx), dtype=bool)
    for i in range(dim):
        mask_i = (zyx[:, i]>=0) & (zyx[:, i]<=region.shape[i]-1)
        mask_shape = mask_shape & mask_i

    # mask for clipped points inside region
    # clip so that region[idx] is valid
    idx_clip = tuple(
        np.clip(zyx[:, i], 0, region.shape[i]-1).astype(int)
        for i in range(3)
    )
    mask_clip = np.asarray(region, dtype=bool)[idx_clip]

    # combine masks
    mask = mask_shape & mask_clip
    return mask

def orients_absdiff(orient1, orient2):
    """ Absolute differences between two orientation arrays.
 
    This function duplicates that in imgutil, to remove dependencies between these two submodules.
    dO = mod(orient2-orient1,pi), then wrapped to [0,pi/2) by taking values>pi/2 to pi-values.

    Args:
        orient1, orient2 (np.ndarray): Two orientations, with values in (-pi/2,pi/2)+n*pi.
    
    Returns:
        dO (np.ndarray): Absolute difference, with the same shape as orient1 (or orient2).
    """
    dO = np.mod(orient2-orient1, np.pi)
    dO = np.where(dO <= np.pi/2, dO, np.pi-dO)
    return dO

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


#=========================
# normals
#=========================

def normals_gen_ref(zyx, dist=None):
    """ Generate default reference point for the directions of normals.

    First calculate the PC1 of all points.
    Cross multiply PC1 with z+ to get a direction.
    The reference point is shifted from the center along the direction by some distance.
    The default distance is the norm of the span of points in all directions.

    Args:
        zyx (np.ndarray): 3d points, with shape=(npts,3).
        dist (float or None): Distance from the center.

    Returns:
        zyx_ref (np.ndarray): Reference point, [z_ref,y_ref,x_ref].
    """
    # setup dist
    if dist is None:
        dist = np.linalg.norm(np.ptp(zyx, axis=0))

    # get pc1 vector
    pca = decomposition.PCA().fit(zyx)
    e1 = pca.components_[0]
    e1 = e1 / np.linalg.norm(e1)

    # direction: cross product of pc1 and z+
    e2 = np.array([1, 0, 0])
    direction = np.cross(e1, e2)

    # reference point
    center = np.mean(zyx, axis=0)
    zyx_ref = center + dist*direction
    return zyx_ref

def normals_pointcloud(pcd, sigma, pt_ref=None):
    """ Estimate normals for point cloud.

    Normals are oriented consistently. If a reference point (pt_ref) is provided, then normals point from pt_ref to pcd.

    Args:
        pcd (open3d.geometry.PointCloud): open3d pointcloud
        sigma (float): Neighborhood radius for normal estimation.
        pt_ref (np.ndarray, optional): Reference point 'inside'. Formatted to be comparable with pcd.points[0].
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
    
    Wraps pcdutil.normals_pointcloud.

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
        hull (sp.spatial.ConvexHull): The convex hull.
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
        hull (sp.spatial.ConvexHull): The convex hull.

    Returns:
        mask (np.ndarray): Array bools indicating whether each point is in the hull, shape=(npts,).
            pts[mask] gives points in the hull.

    References:
        method: https://stackoverflow.com/questions/31404658/check-if-points-lies-inside-a-convex-hull
    """
    eps = np.finfo(np.float64).eps
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1:]
    mask = np.all(np.asarray(pts) @ A.T + b.T < eps, axis=1)
    return mask


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

    # max nearest neighbor distance except self
    pts = np.asarray(mesh_div.vertices)
    kdtree = sp.spatial.KDTree(pts)
    dists, _ = kdtree.query(pts, k=2, p=1, workers=-1)
    dists = dists[:, 1]
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
# graph
#=========================

def neighbors_graph(pts, r_thresh=1, orients=None):
    """ Construct neighbors graph from points.
    
    Vertices represent points.
    Edge is generated when the distance between two points <= r_thresh.
    The orientation of each point can be provided for additional info.
    Given the output graph, edge weights or other metrics can be computed.
    
    Attributes of the output graph g:
        g.vs["coords"]: copies pts, the coordinate of each vertex.
        g.vs["orients"]: copies orients (if provided), the orientation at each point.
        g.es["dist"]: distance between points.
        g.es["dorients"]: orientational difference between two points, in [0,pi/2], (if orients is provided).

    Args:
        pts (np.ndarray): Points with shape=(npts,dim).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        orients (np.ndarray): Orientation at each point, ranged in [0,pi/2], shape=(npts,).
    
    Returns:
        g (igraph.Graph): The neighbor graph.
    """
    # get pairs within r_thresh
    kdtree = sp.spatial.KDTree(pts)
    edges = kdtree.query_pairs(r=r_thresh)
    edges = np.asarray(tuple(edges))

    # distance in each pair
    e1, e2 = np.transpose(edges)
    dr = np.linalg.norm(pts[e1]-pts[e2], axis=1)
    
    # make graph
    g = igraph.Graph()
    g.add_vertices(len(pts))
    g.add_edges(edges)
    g.vs["coords"] = pts
    g.es["dist"] = dr

    # if orientations are provided, add orientational differences
    if orients is not None:
        dO = orients_absdiff(orients[e1], orients[e2])
        g.vs["orients"] = orients
        g.es["dorients"] = dO
    return g

def graph_components(g, n_keep=None):
    """ Extract n_keep largest components in the graph. An iterator.

    Args:
        g (igraph.Graph): The neighbor graph.
        n_keep (int): The number of components to keep.
        
    Yields:
        (size_i, gsub_i): Component i's size, subgraph.
    """
    # get components
    comps = g.components(mode="weak")

    # count
    df = pd.Series(comps.membership).value_counts(
        sort=True, ascending=False
    ).to_frame("size").reset_index()

    # iteration
    for item in df.iloc[:n_keep].itertuples():
        size_i = item.size
        gsub_i = comps.subgraph(item.index)
        yield (size_i, gsub_i)

def neighboring_components(pts, r_thresh=1, n_keep=None):
    """ Extract n_keep largest components in the neighbors graph. An iterator.

    Args:
        pts (np.ndarray): Points with shape=(npts,dim).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        n_keep (int): The number of components to keep.
        
    Yields:
        (size_i, pts_i, gsub_i): Component i's size, points, subgraph.
    """
    # construct graph
    g = neighbors_graph(pts, r_thresh=r_thresh)

    # iteration
    for size_i, gsub_i in graph_components(g, n_keep):
        pts_i = np.asarray(gsub_i.vs["coords"])
        yield (size_i, pts_i, gsub_i)

#=========================
# sorting
#=========================

def sort_pcds_by_ref(pts_arr, pt_ref):
    """ Sort a list of pointclouds by their min distance to a reference point in ascending order.

    Args:
        pts_arr (list of np.ndarray): A list of pointcloud, each with shape=(nptsi,dim).
        pt_ref (np.ndarray): The reference point, shape=(dim,).

    Returns:
        pts_sorted (list of np.ndarray): The sorted list of pointclouds.
    """
    # compare components' distance to ref
    pt_ref = np.asarray(pt_ref).reshape((1, -1))
    dist_arr = [
        np.sum((pts-pt_ref)**2, axis=1).min()
        for pts in pts_arr
    ]

    # sort index, get points
    idx_sorted = np.argsort(dist_arr)
    pts_sorted = [pts_arr[i] for i in idx_sorted]

    return pts_sorted

def sort_pts_by_guide_2d(pts, guide):
    """ Sort 2d points by guiding line.

    Guide line points are assumed sorted.
    Sort the points first by the index of their nearest guide then by the distance.
    Return index of sorted points.

    Args:
        pts (np.ndarray): Points to be sorted, with shape=(npts,dim).
        guide (np.ndarray): Guideline points which are sorted, with shape=(npts_guide,dim).

    Returns:
        idx_pts (np.ndarray): Index of sorted points, with shape=(npts,).
            pts[idx_pts] gives sorted points.
    """
    # setup array formats
    pts = np.asarray(pts, dtype=int)
    guide = points_deduplicate(guide)

    # query dist and idx2guide using KDTree
    kdtree = sp.spatial.KDTree(guide)
    dist, idx_guide = kdtree.query(pts, k=1, workers=-1)

    # sort by first the guide index then the distance
    df = pd.DataFrame(
        data=np.transpose([
            np.arange(len(pts)),
            idx_guide,
            dist
        ]),
        columns=["idx_pts", "idx_guide", "dist"]
    )
    df = df.sort_values(by=["idx_guide", "dist"])

    # get the sorted index for points
    idx_pts = df["idx_pts"].values.astype(int)
    return idx_pts

def sort_pts_by_guide_3d(zyx, guide):
    """ Sort 3d points by guidelines slice by slice.

    Guide line points are assumed sorted in each xy-slice.
    Sort the points in each slice first by the index of their nearest guide then by the distance.
    Return sorted points.

    Args:
        zyx (np.ndarray): 3d points, with shape=(npts,3). Each point is [zi,yi,xi].
        guide (np.ndarray): 3d guideline points sorted in each slice, with shape=(npts_guide,3). Each point is [zi,yi,xi].

    Returns:
        zyx_sorted (np.ndarray): Points sorted in each slice, with shape=(npts,3).
    """
    # setup array formats
    zyx = np.asarray(zyx, dtype=int)
    guide = np.asarray(guide, dtype=int)

    # sort points slice by slice
    zyx_sorted = []
    for z in sorted(np.unique(zyx[:, 0])):
        zyx_i = zyx[zyx[:, 0]==z]
        guide_i = guide[guide[:, 0]==z]
        idx_i = sort_pts_by_guide_2d(zyx_i, guide_i)
        zyx_sorted.append(zyx_i[idx_i])
    
    # collect slices
    zyx_sorted = np.concatenate(zyx_sorted, axis=0)
    return zyx_sorted
