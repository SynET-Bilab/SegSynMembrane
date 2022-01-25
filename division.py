#!/usr/bin/env python
""" trace
"""

import multiprocessing.dummy
import numpy as np
import scipy as sp
import sklearn.decomposition
import skimage
from synseg import utils

__all__ = [
    "Segmentalize",
    "divide_connected"
]


class Segmentalize:
    """ binary image -> segments
    """
    def __init__(self, B, O, r_thresh, max_size=np.inf):
        # save variables
        self.B = B
        self.O = np.copy(O)
        self.nz = B.shape[0]
        self.max_size = max_size
        self.r_thresh = r_thresh
        
        # setup direction
        self.pca = None
        self.set_direction()

        # prep for finding next yx
        self.next_yx_tools = self.prep_find_next_yx()

        # prep disk for dilation
        self.disk = skimage.morphology.disk(self.r_thresh)
    
    def set_direction(self):
        """ set pca for projected yx; align O to pc axis
        """
        # calc pca
        zyx = utils.mask_to_coord(self.B)
        self.pca = sklearn.decomposition.PCA(n_components=1)
        self.pca.fit(zyx[:, 1:])
        
        # align O with pc axis 1
        axis_y, axis_x = self.pca.components_[0]
        inner_prod = np.cos(self.O)*axis_x + np.sin(self.O)*axis_y
        self.O[(self.B>0)&(inner_prod<0)] += np.pi

    def prep_find_next_yx(self):
        """ prep for finding next yx
        :return: {map_dydx, bins}
        """
        # dict from direction to dydx
        # key = index in histogram
        # value = dydx for d_curr, and neighbors as candidates
        map_dydx = {
            0: [(0, 1), (1, 1), (-1, 1)],  # (pi*0/8 ,pi*1/8)
            1: [(1, 1), (0, 1), (1, 0)],  # (pi*1/8 ,pi*2/8)
            2: [(1, 1), (1, 0), (0, 1)],  # (pi*2/8 ,pi*3/8)
            3: [(1, 0), (1, 1), (1, -1)],  # (pi*3/8 ,pi*4/8)
            4: [(1, 0), (1, -1), (1, 1)],  # (pi*4/8 ,pi*5/8)
            5: [(1, -1), (1, 0), (0, -1)],  # (pi*5/8 ,pi*6/8)
            6: [(1, -1), (0, -1), (1, 0)],  # (pi*6/8 ,pi*7/8)
            7: [(0, -1), (1, -1), (-1, -1)],  # (pi*7/8 ,pi*8/8)
            8: [(0, -1), (-1, -1), (1, -1)],  # (pi*8/8 ,pi*9/8)
            9: [(-1, -1), (0, -1), (-1, 0)],  # (pi*9/8 ,pi*10/8)
            10: [(-1, -1), (-1, 0), (0, -1)],  # (pi*10/8 ,pi*11/8)
            11: [(-1, 0), (-1, -1), (-1, 1)],  # (pi*11/8 ,pi*12/8)
            12: [(-1, 0), (-1, 1), (-1, -1)],  # (pi*12/8 ,pi*13/8)
            13: [(-1, 1), (-1, 0), (0, 1)],  # (pi*13/8 ,pi*14/8)
            14: [(-1, 1), (0, 1), (-1, 0)],  # (pi*14/8 ,pi*15/8)
            15: [(0, 1), (-1, 1), (1, 1)],  # (pi*15/8 ,pi*16/8)
        }
        bins = np.pi*np.arange(0, 16.1, 1)/8
        return dict(map_dydx=map_dydx, bins=bins)

    def find_next_yx(self, d_curr):
        """ find next yx candidates according to current direction
        :param d_curr: current direction
        :return: dydx_candidates
            dydx_candidates: three possible dydx's among 8 neighbors
        """
        # direction: keep in range (0, 2pi)
        d_curr = np.mod(d_curr, 2*np.pi)

        # categorize: into bins from
        loc_bin = np.histogram(d_curr, bins=self.next_yx_tools["bins"])[0].argmax()

        # obtain candidates
        dydx_candidates = self.next_yx_tools["map_dydx"][loc_bin]
        return dydx_candidates

    def find_next_direction(self, d_curr, o_next):
        """ convert next orientation to direction that's closest to current direction
        :param d_curr: current direction
        :param o_next: next orientation
        :return: d_next
        """
        # diff if d_next=o_next
        diff1 = np.mod(o_next-d_curr, 2*np.pi)
        diff1 = diff1 if diff1 <= np.pi else diff1-2*np.pi
        # diff if d_next=o_next+pi
        diff2 = np.mod(o_next+np.pi-d_curr, 2*np.pi)
        diff2 = diff2 if diff2 <= np.pi else diff2-2*np.pi

        # pick one with smaller diff
        diff = diff1 if np.abs(diff1) <= np.abs(diff2) else diff2
        d_next = d_curr + diff
        return d_next

    def trace_connected(self, yx_start, map_yxd):
        """ trace segment from current (y,x) in one direction
        :param yx_start: current (y,x)
        :param yx_trace: list of (y,x)'s in the trace
        :param map_yxd: {(y,x): direction}
        """
        yx_trace = []
        d_trace = []
        # BFS
        yx_tovisit = [yx_start]
        while len(yx_tovisit) > 0:
            yx_curr = yx_tovisit[0]

            # continue: yx_curr is not in image, or visited
            if (yx_curr not in map_yxd) or (map_yxd[yx_curr] is None):
                yx_tovisit.remove(yx_curr)
                continue

            # visit yx_curr, flag with None
            d_curr = map_yxd[yx_curr]
            yx_trace.append(yx_curr)
            d_trace.append(d_curr)
            map_yxd[yx_curr] = None
            yx_tovisit.remove(yx_curr)
            
            # break: if max_size is reached
            if len(yx_trace) >= self.max_size:
                break

            # update yx_tovisit
            for dydx in self.find_next_yx(d_curr):
                yx_next = (yx_curr[0]+dydx[0], yx_curr[1]+dydx[1])
                # align next orientation, update dict
                if (yx_next in map_yxd) and (map_yxd[yx_next] is not None):
                    d_next = self.find_next_direction(d_curr, map_yxd[yx_next])
                    map_yxd[yx_next] = d_next
                    yx_tovisit.append(yx_next)

        return yx_trace, d_trace
            
    def segment2d_connected(self, Bc, Oc):
        """ segmentalize a connected component
        :param Bc: 2d binary image of the component
        :param Oc: 2d orientation image of the component
        :return: yx_traces, d_traces
            yx_traces: [yx_comp_0, yx_comp_1, ...]
            d_traces: [d_comp_0, d_comp_1, ...]
        """
        # get yx and d from image, sort by pc
        pos = np.nonzero(Bc)
        idx_sort = np.argsort(self.pca.transform(np.transpose(pos))[:, 0])
        yx_sort = np.stack([pos[i][idx_sort] for i in range(2)], axis=1)
        yx_sort = [tuple(yx_i) for yx_i in yx_sort]
        d_sort = Oc[pos][idx_sort]
        map_yxd = dict(zip(yx_sort, d_sort))

        # trace until all yx's are exhausted
        yx_traces = []
        d_traces = []
        while len(yx_sort) > 0:
            # trace
            yx_trace_i, d_trace_i = self.trace_connected(yx_sort[0], map_yxd)
            # record
            yx_traces.append(yx_trace_i)
            d_traces.append(d_trace_i)
            # remove traced yx's
            for yx in yx_trace_i:
                yx_sort.remove(yx)
        
        return yx_traces, d_traces

    def segment3d(self):
        """ segmentalize 3d binary image self.B
        :return: L, iz_segs, o_segs
            L: 3d labeled image, label starts from 1
            iz_segs: iz of each label, 1d array with n_label elements
            o_segs: orientation of each label
        """
        # segmentalize each slice
        def calc_one(iz):
            yxd_iz = []
            for _, Bzc in utils.extract_connected(self.B[iz]):
                yx_zc, d_zc = self.segment2d_connected(Bzc, self.O[iz])
                for yx_i, d_i in zip(yx_zc, d_zc):
                    yxd_iz.append((iz, yx_i, np.mean(d_i)))
            return yxd_iz
        
        # flatten array
        yxd = []
        for yxd_iz in map(calc_one, range(self.nz)):
            yxd.extend(yxd_iz)

        # compile into 3d image
        L = np.zeros(self.B.shape, dtype=np.int_)
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
        B_arr: [B_0, B_1, ...], binary image for each cluster
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
    
    # label image
    n_segs = len(iz_segs)
    label_segs = np.arange(n_segs) + 1
    B_arr = []
    for i in range(2):
        B_i = np.zeros_like(L)
        mask_i = np.isin(L, label_segs[clust.labels_ == i])
        B_i[mask_i] = 1
        B_arr.append(B_i)
    return B_arr
