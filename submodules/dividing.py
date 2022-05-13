
import numpy as np
import sklearn.cluster
import pandas as pd
from etsynseg import imgutils, pcdutils

__all__ = [
    "divide_spectral", "divide_two_auto"
]

def divide_spectral(zyx, orients, r_thresh, sigma_dO=np.pi/4, n_clusts=2):
    """ Divide connected pixels into clusters using Leiden and spectral methods.
    
    The constructed graph is weighted by orientational differences.
        weight(dO) = exp(-0.5*(dO/sigma_dO)**2)
    Leiden clustering is applied to aggregate the graph of points and reduce the number of nodes.
    Spectral clustering is applied to cut the aggregated graph into n_clusts parts.

    Args:
        zyx (np.ndarray): Points with shape=(npts, ndim) and in format [[zi,yi,xi],...].
        orients (np.ndarray): Orientation at each point, ranged in [0,pi/2], shape=(npts,).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        sigma_dO (float): Sigma of orientational difference for edge weighting.
        n_clusts (int): The number of output clusters.

    Returns:
        zyx_clusts (list of np.ndarray): Points for each cluster, sorted from the largest in size.
            [zyx1,zyx2,...], where zyxi are points with shape=(npts,3) for cluster i.
    """
    # build graph for points
    g_pts = pcdutils.neighbors_graph(
        zyx, orients=orients, r_thresh=r_thresh
    )
    dorients = np.asarray(g_pts.es["dorients"])
    g_pts.es["weights"] = np.exp(-0.5*(dorients/sigma_dO)**2)

    # partition points using leiden
    community = g_pts.community_leiden(
        objective_function="modularity",
        weights=g_pts.es["weights"]
    )

    # aggregate points to get graph for groups
    g_grps = community.cluster_graph(
        combine_edges={"weights": "sum"}
    )
    mat_grps = g_grps.get_adjacency_sparse(attribute="weights")

    # spectral clustering on groups
    clust_grps = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusts,
        affinity="precomputed",
        assign_labels="discretize"
    )
    clust_grps.fit(mat_grps)
    
    # assign cluster label to each point
    # membership[i] gives group index (i_grp) of point i
    # clust_grps.labels_[i_grp] gives cluster index of group i_grp
    # then label_pts[i] is the cluster index of points i
    label_pts = clust_grps.labels_[community.membership]

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
