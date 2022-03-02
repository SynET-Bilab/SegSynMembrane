
import numpy as np
from scipy import spatial
import igraph
import leidenalg
import sklearn.cluster
import pandas as pd
from etsynseg import utils

__all__ = [
    "divide_spectral"
]

def divide_spectral(zyx, group_rthresh, group_size, n_clusters=2):
    """ divide connected image into n_clusters clusters. first use leiden partition to group points, then use spectral clustering to get n_clusters clusters
    :param zyx: points
    :param group_rthresh: radius for building nn-graph for points
    :param group_size: max size of groups, < cleft width
    :param n_clusters: number of clusters to divide into
    :return: zyx_clusts
        zyx_clusts: [zyx_1, zyx_2, ...], points for each cluster, sorted from the largest in size
    """
    # build nn-graph for points
    kdtree = spatial.KDTree(zyx)
    pairs = kdtree.query_pairs(r=group_rthresh)
    g_pts = igraph.Graph()
    g_pts.add_vertices(len(zyx))
    g_pts.add_edges(pairs)

    # partition using leiden
    part_pts = leidenalg.find_partition(
        g_pts,
        leidenalg.ModularityVertexPartition,
        max_comm_size=int(group_size)
    )

    # aggregate to get graph for groups
    part_grps = part_pts.aggregate_partition()
    mat_grps = part_grps.graph.get_adjacency_sparse()

    # spectral clustering on groups
    clust_grps = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters,
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

    # output clusters in the order of size
    zyx_clusts = []
    for i in label_clusts:
        zyx_clusts.append(zyx[label_pts==i])

    return zyx_clusts

def divide_to_two(zyx, group_rthresh, group_size, ratio_comps=0.5, maxiter=10, zfilter=-1):
    """ divide into two parts, until size ratio > ratio_comps
    :param zyx: points
    :param group_rthresh: radius for building nn-graph for points
    :param group_size: max size of groups, < cleft width
    :param ratio_comps: divide the largest component if size2/size1<ratio_comps
    :param maxiter: max iteration for division
    :param zfilter: sets min dz for components, dzmin=dzspan*zfilter
    :return: zyx_comps
        zyx_comps: [zyx1, zyx2]
    """
    # setup
    # shape
    shape = np.ceil(np.max(zyx, axis=0)).astype(np.int_) + 1
    # dzmin
    dzfull = np.ptp(zyx[:, 0])
    if zfilter <= 0:
        dzmin = dzfull + zfilter
    elif zfilter < 1:
        dzmin = int(dzfull*zfilter)
    else:
        dzmin = int(zfilter)
    
    # extract connected from points
    def extract_connected(zyx_i, n_keep):
        B_c = utils.extract_connected(
            utils.points_to_voxels(zyx_i, shape),
            n_keep=n_keep, connectivity=3
        )
        zyx_c = [utils.voxels_to_points(comp[1]) for comp in B_c]
        return zyx_c

    # whether need to further divide
    def need_to_divide(zyx_comps):
        # only one component: True
        if len(zyx_comps) == 1:
            return True
        else:
            zyx1, zyx2 = zyx_comps[:2]
            # size of component-2 too small: True
            if len(zyx2)/len(zyx1) < ratio_comps:
                return True
            # z-span of component-2 too small: True
            # loophole here: z-span of component-1 could be too small
            elif np.ptp(zyx2[:, 0]) < dzmin:
                return True
            else:
                return False

    # extract two components
    zyx_comps = extract_connected(zyx, 2)

    # if two membranes seem connected, divide
    iiter = 0
    while need_to_divide(zyx_comps) or (iiter >= maxiter):
        iiter += 1
        zyx_comps_raw = divide_spectral(
            zyx_comps[0], group_rthresh, group_size, n_clusters=2
        )
        # get connected
        zyx_comps = [
            extract_connected(zyx_i, 1)[0]
            for zyx_i in zyx_comps_raw
        ]
    
    return zyx_comps

