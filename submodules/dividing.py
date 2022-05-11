
import numpy as np
import scipy as sp
import igraph
import leidenalg
import sklearn.cluster
import pandas as pd
from etsynseg import imgutils, pcdutils

__all__ = [
    "weighted_graph_orient", "weight_matrix_agg",
    "divide_spectral", "divide_two_auto"
]


def weighted_graph_orient(zyx, O, r_thresh, sigma_dO=np.pi/6):
    """ Generated graph whose vertices are points and edges are weighted by orientational differences.

    Weight=np.exp(-0.5*(dO/sigma_dO)**2) for all pairs with distance<r_thresh 

    Args:
        zyx (np.ndarray): Points with shape=(npts, ndim) and in format [[zi,yi,xi],...].
        O (np.ndarray): Orientation, with shape=(nz,ny,nx).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        sigma_dO (float): Sigma of orientational difference for edge weighting.
    
    Returns:
        g (igraph.Graph): Graph with weighted edges (g.es["weight]).
    """
    zyx = np.asarray(zyx, dtype=int)

    # get pairs within r_thresh
    kdtree = sp.spatial.KDTree(zyx)
    pairs = kdtree.query_pairs(r=r_thresh)
    pairs = tuple(pairs)

    # calc orientational difference between pairs
    Oflat = O[tuple(zyx.T)]  # flatten O
    idx1, idx2 = np.transpose(pairs)
    dO = imgutils.absdiff_orient(Oflat[idx1], Oflat[idx2])

    # make graph
    g = igraph.Graph()
    g.add_vertices(len(zyx))
    g.add_edges(pairs)
    g.es["weight"] = np.exp(-0.5*(dO/sigma_dO)**2)
    return g

def weight_matrix_agg(g_agg):
    """ Generate edge weight matrix for aggregated graph.

    The aggregated graph is produced by leidenalg partition.aggregate_partition().graph.
    This graph contains self-edges and repeated edges.
    In the weight matrix, self-edges are excluded, repeated edges are weight-summed.

    Args:
        g_agg (igraph.Graph): The aggregated graph.
    
    Returns:
        weight_mat (sp.sparse.csr_matrix): Weight matrix.
    """
    # get edges
    e_dict = {}
    ew_arr = []
    e1_arr = []
    e2_arr = []
    for e in g_agg.es:
        e1, e2 = e.tuple
        # exclude self-edges
        if e1 == e2:
            continue
        # add new edges
        if e.tuple not in e_dict:
            ew_arr.append(e["weight"])
            e1_arr.append(e1)
            e2_arr.append(e2)
            e_dict[e.tuple] = len(ew_arr) - 1
        # add weight to existed edges
        else:
            eid = e_dict[e.tuple]
            ew_arr[eid] += e["weight"]

    # generate weight matrix
    # symmetrize
    value = ew_arr+ew_arr
    row = e1_arr+e2_arr
    col = e2_arr+e1_arr
    # construct sparse matrix
    nv = g_agg.vcount()
    weight_mat = sp.sparse.csr_matrix(
        (value, (row, col)),
        shape=(nv, nv)
    )
    return weight_mat

def divide_spectral(zyx, O, r_thresh, sigma_dO, max_group_size=0, n_clust=2):
    """ Divide connected pixels into clusters using Leiden and spectral methods.
    
    The constructed graph is weighted by orientational differences.
    Leiden clustering is applied to aggregate the graph of points and reduce the number of nodes.
    Spectral clustering is applied to cut the aggregated graph into n_clust parts.

    Args:
        zyx (np.ndarray): Points with shape=(npts, ndim) and in format [[zi,yi,xi],...].
        O (np.ndarray): Orientation, with shape=(nz,ny,nx).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        sigma_dO (float): Sigma of orientational difference for edge weighting.
        max_group_size (int): Max size of Leiden clusters. 0 for no-limit.
        n_clust (int): The number of output clusters.

    Returns:
        zyx_clusts (list of np.ndarray): Points for each cluster, sorted from the largest in size.
            [zyx1,zyx2,...], where zyxi are points with shape=(npts,3) for cluster i.
    """
    # build graph for points
    g_pts = weighted_graph_orient(zyx, O, r_thresh=r_thresh, sigma_dO=sigma_dO)

    # partition points using leiden
    part_pts = leidenalg.find_partition(
        g_pts,
        leidenalg.ModularityVertexPartition,
        weights=g_pts.es["weight"],
        max_comm_size=max_group_size
    )

    # aggregate points to get graph for groups
    part_grps = part_pts.aggregate_partition()
    mat_grps = weight_matrix_agg(part_grps.graph)

    # spectral clustering on groups
    clust_grps = sklearn.cluster.SpectralClustering(
        n_clust=n_clust,
        affinity="precomputed",
        assign_labels="discretize"
    )
    clust_grps.fit(mat_grps)
    
    # assign cluster label to each point
    # clust_grps.labels_[i_grp] gives cluster index for group with index i_grp
    # use part_pts.membership to relate i_grp to points positions
    label_pts = clust_grps.labels_[part_pts.membership]

    # sort cluster labels by size
    label_clusts = list(
        pd.Series(clust_grps.labels_)
        .value_counts(sort=True, ascending=False).index
    )

    # output clusters in the order of decreasing size
    zyx_clusts = []
    for i in label_clusts:
        zyx_clusts.append(zyx[label_pts==i])

    return zyx_clusts

def divide_two_auto(zyx, O, r_thresh, sigma_dO, size_ratio=0.5, max_iter=10, min_zspan=-1):
    """ Divide connected pixels into two clusters.

    Apply divide_spectral iteratively, until criteria is met.

    Args:
        zyx (np.ndarray): Points with shape=(npts, ndim) and in format [[zi,yi,xi],...].
        O (np.ndarray): Orientation, with shape=(nz,ny,nx).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        sigma_dO (float): Sigma of orientational difference for edge weighting.
        size_ratio (float): Divide the largest component if size2/size1<size_ratio.
        max_iter (int): Max number of iterations for division.
        min_zspan (int): Min span in z for components. -1 for full span from 0 to nz-1.

    Returns:
        zyx_clusts (list of np.ndarray): Points for each cluster, sorted from the largest in size.
            [zyx1,zyx2,...], where zyxi are points with shape=(npts,3) for cluster i.
    """
    # setup
    zyx = np.round(zyx).astype(int)
    shape = np.max(zyx, axis=0).astype(int) + 1
    if min_zspan <= 0:
        min_zspan = np.ptp(zyx[:, 0])
    
    # extract connected from points
    def connected_points(zyx_i, n_keep):
        B_c = imgutils.connected_components(
            pcdutils.points2pixels(zyx_i, shape),
            n_keep=n_keep, connectivity=3
        )
        zyx_c = [pcdutils.pixels2points(comp[1]) for comp in B_c]
        return zyx_c

    # whether need to further divide
    def need_to_divide(zyx_comps):
        # only one component: True
        if len(zyx_comps) == 1:
            return True
        # case of two components
        else:
            zyx1, zyx2 = zyx_comps[:2]
            # size of component-2 too small: True
            if len(zyx2)/len(zyx1) < size_ratio:
                return True
            # z-span of component-2 too small: True
            # loophole here: z-span of component-1 could be too small
            elif np.ptp(zyx2[:, 0]) < min_zspan:
                return True
            else:
                return False

    # extract two components
    zyx_comps = connected_points(zyx, 2)

    # if two membranes seem connected, divide
    i_iter = 0
    while need_to_divide(zyx_comps) and (i_iter <= max_iter):
        i_iter += 1
        zyx_comps_raw = divide_spectral(
            zyx_comps[0], O,
            r_thresh=r_thresh, sigma_dO=sigma_dO,
            n_clust=2
        )
        # get connected, to avoid spectral clustering of disconnected graphs
        zyx_comps = [
            connected_points(zyx_i, 1)[0]
            for zyx_i in zyx_comps_raw
        ]
    
    return zyx_comps

# def divide_two_mask(zyx, zyx_mod_divide, zyx_bound, shape, r_dilate=1):
#     """ divide into two parts using masks
#         zyx: points to be divided
#         zyx_mod_divide: zyx of the model for dividing
#         zyx_bound: points for boundary
#         shape: (nz,ny,nx)
#         r_dilate: radius for dilation of mask for dividing
#     Returns: zyx_comps
#         zyx_comps: [zyx1, zyx2]
#     """
#     # construct mask for division
#     mask_divide = io.model_to_mask(
#         zyx_mod_divide, shape=shape,
#         closed=False, extend=True, amend=False
#     )
#     mask_divide = skimage.morphology.binary_dilation(
#         mask_divide,
#         skimage.morphology.ball(r_dilate)
#     )

#     # construct mask for boundary
#     mask_bound = utils.points2pixels(zyx_bound, shape)

#     # divide mask_bound into two parts
#     mask_comps = [
#         comp[1] for comp in
#         imgutils.connected_components(
#             mask_bound*(1-mask_divide),
#             n_keep=2, connectivity=1)
#     ]

#     # mask points in the two parts
#     B = utils.points2pixels(zyx, shape)
#     zyx_comps = [
#         utils.pixels2points(B*mask)
#         for mask in mask_comps
#     ]
#     return zyx_comps
