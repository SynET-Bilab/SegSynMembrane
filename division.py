""" trace
"""

import multiprocessing.dummy
import numpy as np
import scipy as sp
import pandas as pd
import sklearn.decomposition
import sklearn.cluster
import skimage
from etsynseg import utils, trace

__all__ = [
    "Segmentalize",
    "divide_connected"
]


class Segmentalize:
    """ binary image -> segments
    """
    def __init__(self, B, O, r_thresh, max_size=np.inf):
        # save variables
        self.shape = B.shape
        self.nz = B.shape[0]
        
        # setup trace
        self.tracing = trace.Trace(B, O, max_size=max_size)

        # prep disk for dilation
        self.r_thresh = r_thresh
        self.disk = skimage.morphology.disk(self.r_thresh)

    def segment3d(self):
        """ segmentalize 3d binary image self.B
        :return: L, iz_segs, o_segs
            L: 3d labeled image, label starts from 1
            iz_segs: iz of each label, 1d array with n_label elements
            o_segs: orientation of each label
        """
        # segmentalize into segments
        yxd = []
        for iz in range(self.nz):
            yx_iz, d_iz = self.tracing.bfs2d(iz)
            for yx_i, d_i in zip(yx_iz, d_iz):
                yxd.append((iz, yx_i, np.mean(d_i)))

        # compile into 3d image
        L = np.zeros(self.shape, dtype=np.int_)
        iz_segs = np.zeros(len(yxd), dtype=np.int_)
        d_segs = np.zeros(len(yxd), dtype=np.float_)
        for i, (iz, yx_i, d_i) in enumerate(yxd):
            label_i = i + 1
            idx_yx = tuple(np.transpose(yx_i))
            L[iz][idx_yx] = label_i
            iz_segs[i] = iz
            d_segs[i] = d_i
        o_segs = np.mod(d_segs, np.pi)
        return L, iz_segs, o_segs
            
    def pairwise_weight(self, L, iz_segs, o_segs, n_proc=None):
        """ calculate pairwise weight between segments
        :param L, iz_segs, o_segs: results from segment3d
        :param n_proc: number of processors for multithreading
        :return: mat
            mat: symmetric csr_matrix of weight, diagonal=1
        """
        # weight between one segment and its neighbors
        def calc_one(label):
            # get info for this segment
            idx = label - 1
            iz = iz_segs[idx]
            o = o_segs[idx]
            
            # calc overlap with other segments
            mask = skimage.morphology.binary_dilation(L[iz]==label, self.disk)
            overlap = L[iz:iz+self.r_thresh, mask]
            label_nbs = np.unique(overlap[overlap>0])
            # only count labels_nbs>label
            label_nbs = label_nbs[label_nbs>label]
            
            # calc weight
            idx_nbs = label_nbs - 1
            o_nbs = o_segs[idx_nbs]
            weight = 1 - utils.absdiff_orient(o_nbs, o)*2/np.pi

            # return
            data = np.concatenate(([1], weight, weight))
            i_ext = idx*np.ones(len(label_nbs), dtype=np.int_)
            row = np.concatenate(([idx], i_ext, idx_nbs))
            col = np.concatenate(([idx], idx_nbs, i_ext))
            return data, row, col
        
        # info
        n_segs = len(iz_segs)
        labels = np.arange(n_segs)+1

        # iterate for each segment
        pool = multiprocessing.dummy.Pool(n_proc)
        result = pool.map(calc_one, labels)
        pool.close()

        # concaternate results
        data = []
        row = []
        col = []
        for data_i, row_i, col_i in result:
            data.append(data_i)
            row.append(row_i)
            col.append(col_i)
        data = np.concatenate(data)
        row = np.concatenate(row)
        col= np.concatenate(col)

        # make sparse matrix
        mat = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(n_segs, n_segs)
        )
        return mat

def divide_connected(B, O, seg_max_size=10, seg_neigh_thresh=5, n_clusters=2):
    """ divide connected image into n_clusters parts
    :param B, O: binary image, orientation
    :param seg_max_size: max size of each segment, < cleft width
    :param seg_neigh_thresh: distance threshold for finding neighbors
    :param n_clusters: number of clusters to divide into
    :return: B_arr
        B_arr: [B_0, B_1, ...], binary image for each cluster, sorted from largest in size
    """
    # segmentalize image
    segment = Segmentalize(B, O, max_size=seg_max_size, r_thresh=seg_neigh_thresh)
    L, iz_segs, o_segs = segment.segment3d()

    # build affinity matrix
    mat = segment.pairwise_weight(L, iz_segs, o_segs)

    # spectral clustering
    clust = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize"
    )
    clust.fit(mat)
    
    # sort cluster labels by size
    clust_labels = (pd.Series(clust.labels_)
                .value_counts(sort=True, ascending=False)
                .index.tolist())

    # label image
    n_segs = len(iz_segs)
    label_segs = np.arange(n_segs) + 1
    B_arr = []
    for i in clust_labels:
        B_i = np.zeros_like(L)
        mask_i = np.isin(L, label_segs[clust.labels_ == i])
        B_i[mask_i] = 1
        B_arr.append(B_i)
    return B_arr
