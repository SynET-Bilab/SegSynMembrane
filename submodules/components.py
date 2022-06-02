
import numpy as np
import sklearn.cluster
import pandas as pd
from etsynseg import pcdutil, features

__all__ = [
    # division
    "divide_spectral_graph", "divide_spectral_points",
    # components
    "extract_components_one", "extract_components_two", "extract_components_regions"
]


#=========================
# division
#=========================

def divide_spectral_graph(g, n_clusts=2):
    """ Divide a connected graph into subgraphs using Leiden and spectral methods.
    
    The input graph consists of one vertex for each point. Should contain "weight" in edge attributes.
    The edge weights are use for clustering.
    Leiden clustering is applied to aggregate the graph of points and reduce the number of nodes.
    Spectral clustering is applied to cut the aggregated graph into n_clusts parts.

    Args:
        g (igraph.Graph): The input graph.
        n_clusts (int): The number of output clusters.

    Returns:
        gsub_arr (list of igraph.Graph): A list of n_clusts subgraphs, one for each cluster. Subgraphs are sorted by decreasing size (the number of vertices).
    """
    # partition points using leiden
    community = g.community_leiden(
        objective_function="modularity",
        weights=g.es["weight"]
    )

    # aggregate points to get graph for groups
    g_grps = community.cluster_graph(
        combine_edges={"weight": "sum"}
    )
    mat_grps = g_grps.get_adjacency_sparse(attribute="weight")

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
        pd.Series(label_pts).value_counts(
            sort=True, ascending=False
        ).index
    )

    # output clusters in the order of decreasing size
    gsub_arr = []
    for i in label_clusts:
        gsub_i = g.subgraph(np.nonzero(label_pts == i)[0])
        gsub_arr.append(gsub_i)

    return gsub_arr


def divide_spectral_points(zyx, orients, r_thresh, sigma_dO=np.pi/4, n_clusts=2):
    """ Divide neighboring points into clusters using Leiden and spectral methods.
    
    The constructed graph is weighted by orientational differences.
        weight(dO) = exp(-0.5*(dO/sigma_dO)**2)
    Leiden clustering is applied to aggregate the graph of points and reduce the number of nodes.
    Spectral clustering is applied to cut the aggregated graph into n_clusts parts.

    Args:
        zyx (np.ndarray): Points with shape=(npts,dim) and in format [[zi,yi,xi],...].
        orients (np.ndarray): Orientation at each point, ranged in [0,pi/2], shape=(npts,).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        sigma_dO (float): Sigma of orientational difference for edge weighting.
        n_clusts (int): The number of output clusters.

    Returns:
        zyx_clusts (list of np.ndarray): Points for each cluster, sorted from the largest in size.
            [zyx1,zyx2,...], where zyxi are points with shape=(npts,3) for cluster i.
    """
    # build graph for points
    g_pts = pcdutil.neighbors_graph(
        zyx, orients=orients, r_thresh=r_thresh
    )
    dorients = np.asarray(g_pts.es["dorient"])
    g_pts.es["weight"] = np.exp(-0.5*(dorients/sigma_dO)**2)

    # partition into subgraphs
    gsub_arr = divide_spectral_graph(g_pts, n_clusts)

    # output points
    zyx_clusts = [
        np.asarray(g_i.vs["coord"])
        for g_i in gsub_arr
    ]
    return zyx_clusts


#=========================
# components
#=========================

def extract_components_one(zyx, r_thresh=1, min_size=0):
    """ Extract the largest component in the neighboring graph of the points.

    A min_size can be provided.
    If the size of the largest component < min_size, raise error.

    Args:
        zyx (np.ndarray): Points with shape=(npts,dim) and in format [[zi,yi,xi],...].
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        min_size (int): The minimum size of components.
    
    Returns:
        zyx1 (np.ndarray): Points in the largest component. Shape=(npts1,dim).
    """
    size1, zyx1, _ = next(pcdutil.neighboring_components(
        zyx, r_thresh, n_keep=1
    ))
    if size1 < min_size:
        raise RuntimeError(f"The largest component (size={size1}) < min_size ({min_size}).")
    return zyx1

def extract_components_two(zyx, r_thresh=1, orients=None, sigma_dO=np.pi/4, min_size=0):
    """ Extract the largest two components by the neighboring graph of the points.

    A min_size can be provided.
    If the size of the largest component < min_size, raise error.
    If the size of the 2nd largest component < min_size, then divide the largest one into two, until the criteria is met.

    Args:
        zyx (np.ndarray): Points with shape=(npts,dim) and in format [[zi,yi,xi],...].
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        orients (np.ndarray): Orientation at each point, ranged in [0,pi/2], shape=(npts,).
            If None, will calculate using features.
        sigma_dO (float): Sigma of orientational difference for edge weighting.
        mask (np.ndarray, optional): Mask points with shape=(npts_mask,dim).
            zyx-points not in the mask (dist>0) are ignored.
        min_size (int): The minimum size of components.
    
    Returns:
        zyx1, zyx2 (np.ndarray): Points in the largest two component.  Shape=(nptsi,dim), i=1,2.
    """
    # calculate orientation if not provided
    if orients is None:
        orients = features.points_orientation(zyx, sigma=r_thresh)

    # construct neighbors graph
    g = pcdutil.neighbors_graph(
        zyx, r_thresh=r_thresh, orients=orients
    )
    dorients = np.asarray(g.es["dorient"])
    g.es["weight"] = np.exp(-0.5*(dorients/sigma_dO)**2)

    # initial extraction of the larges two components
    comps_iter = pcdutil.graph_components(g, n_keep=2)
    size1, gsub1 = next(comps_iter)
    # case when there is no 2nd largest component
    try:
        size2, gsub2 = next(comps_iter)
    except StopIteration:
        size2 = min(0, min_size-1)

    # setup the test for termination
    def terminate_division(size1, size2):
        if size1 < min_size:
            raise RuntimeError(f"The largest component (size={size1}) < min_size ({min_size}).")
        elif size2 >= min_size:
            return True
        else:
            return False

    # iterative division until the termination criteria is met
    while not terminate_division(size1, size2):
        gsub_arr = divide_spectral_graph(gsub1, n_clusts=2)
        gsub1, gsub2 = gsub_arr[:2]
        size1 = gsub1.vcount()
        size2 = gsub2.vcount()

    # get points from subgraphs
    zyx1 = np.asarray(gsub1.vs["coord"])
    zyx2 = np.asarray(gsub2.vs["coord"])

    return zyx1, zyx2

def extract_components_regions(zyx, region_arr, r_thresh=1, min_size=0):
    """ Extract the largest components in each region.

    Args:
        zyx (np.ndarray): Points with shape=(npts,dim) and in format [[zi,yi,xi],...].
        region_arr (list of np.ndarray): List of regions. Each is a binary image with shape=(nz,ny,nx).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.
        min_size (int): The minimum size of components.

    Returns:
        zyx_arr (list of np.ndarray): List of points in each region. Each has shape (npts_in_region_i,dim).
    """
    zyx_arr = []
    for region_i in region_arr:
        mask_i = pcdutil.points_in_region(zyx, region_i)
        zyx_i = extract_components_one(zyx[mask_i], r_thresh, min_size)
        zyx_arr.append(zyx_i)
    return zyx_arr
