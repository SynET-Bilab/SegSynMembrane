""" trace
"""

import numpy as np
import sklearn.decomposition
from synseg import utils

__all__ = [
    "Trace"
]


class Trace:
    """ trace in 2d planes
    usage:
        tracing = synseg.trace.Trace(B, O)
        yx_traces_iz, d_traces_iz = tracing.bfs2d(iz)
        traces = tracing.bfs3d()
        yx_traces_iz, d_traces_iz = tracing.dfs2d(iz)
        traces = tracing.dfs3d()
    """
    def __init__(self, B, O, max_size=np.inf):
        # save variables
        self.B = B
        self.O = np.copy(O)
        self.nz = B.shape[0]
        self.max_size = max_size
        
        # setup direction
        self.pca = None
        self.set_direction()

        # prep for finding next yx
        self.next_yx_tools = self.prep_find_next_yx()
    
    #=========================
    # auxiliary functions
    #=========================
    
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

    #=========================
    # breadth-first-scan
    #=========================
    
    def bfs2d_from_point(self, yx_start, map_yxd):
        """ trace segment from current (y,x) in one direction
        :param yx_start: current (y,x)
        :param map_yxd: {(y,x): direction}
        :return: yx_trace, d_trace
            yx_trace: [(y,x)_1,(y,x)_2,...]
            d_trace: [d_1,d_2,...]
        """
        yx_trace = []
        d_trace = []

        # BFS
        yx_tovisit = [yx_start]
        while len(yx_tovisit) > 0:
            yx_curr = yx_tovisit[0]
            yx_tovisit = yx_tovisit[1:]

            # continue: if yx_curr is not in image, or visited
            if (yx_curr not in map_yxd) or (map_yxd[yx_curr] is None):
                continue

            # visit yx_curr, flag with None
            d_curr = map_yxd[yx_curr]
            yx_trace.append(yx_curr)
            d_trace.append(d_curr)
            map_yxd[yx_curr] = None
            
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

    def bfs2d(self, iz):
        """ trace a slice
        :param iz: z index of slice
        :return: yx_traces, d_traces
            yx_traces: [yx_trace_0, yx_trace_1, ...]
            d_traces: [d_trace_0, d_trace_1, ...]
        """
        B_iz = self.B[iz]
        O_iz = self.O[iz]

        # get yx and d from image, sort by pc
        pos = np.nonzero(B_iz)
        idx_sort = np.argsort(self.pca.transform(np.transpose(pos))[:, 0])
        yx_sort = np.stack([pos[i][idx_sort] for i in range(2)], axis=1)
        yx_sort = [tuple(yx_i) for yx_i in yx_sort]
        d_sort = O_iz[pos][idx_sort]
        map_yxd = dict(zip(yx_sort, d_sort))

        # trace until all yx's are exhausted
        yx_traces = []
        d_traces = []
        while len(yx_sort) > 0:
            # trace
            yx_trace_i, d_trace_i = self.bfs2d_from_point(yx_sort[0], map_yxd)
            # record
            yx_traces.append(yx_trace_i)
            d_traces.append(d_trace_i)
            # remove traced yx's
            for yx in yx_trace_i:
                yx_sort.remove(yx)
        
        return yx_traces, d_traces

    def bfs3d(self):
        """ trace 3d binary image self.B
        :return: traces
            traces: [(iz_1, yx_traces_1, d_traces_1),...]
        """
        # flatten array
        traces = []
        for iz in range(self.nz):
            yx_iz, d_iz = self.bfs2d(iz)
            traces.append((iz, yx_iz, d_iz))
        return traces

    #=========================
    # depth-first-scan
    #=========================
    
    def dfs2d_from_point(self, yx_curr, map_yxd, yx_trace, d_trace):
        """ trace segment from current (y,x) in one direction
        :param yx_curr: current (y,x)
        :param map_yxd: {(y,x): direction}
        :param yx_trace, d_trace: list of (y,x)'s, d's in the trace
        :return: yx_trace, d_trace
            yx_trace: [(y,x)_1,(y,x)_2,...]
            d_trace: [d_1,d_2,...]
        """
        # return: if yx_curr is not in image, or visited
        if (yx_curr not in map_yxd) or (map_yxd[yx_curr] is None):
            return

        # visit yx_curr, flag with None
        d_curr = map_yxd[yx_curr]
        yx_trace.append(yx_curr)
        d_trace.append(d_curr)
        map_yxd[yx_curr] = None

        # return: if max_size is reached
        if len(yx_trace) >= self.max_size:
            return

        # visit next
        for dydx in self.find_next_yx(d_curr):
            yx_next = (yx_curr[0]+dydx[0], yx_curr[1]+dydx[1])
            # align next orientation, update dict
            if (yx_next in map_yxd) and (map_yxd[yx_next] is not None):
                d_next = self.find_next_direction(d_curr, map_yxd[yx_next])
                map_yxd[yx_next] = d_next
                self.dfs2d_from_point(yx_next, map_yxd, yx_trace, d_trace)
                break
        return

    def dfs2d(self, iz):
        """ trace a slice
        :param iz: z index of slice
        :return: yx_traces, d_traces
            yx_traces: [yx_trace_0, yx_trace_1, ...]
            d_traces: [d_trace_0, d_trace_1, ...]
        """
        B_iz = self.B[iz]
        O_iz = self.O[iz]
        
        # get yx and d from image, sort by pc
        pos = np.nonzero(B_iz)
        idx_sort = np.argsort(self.pca.transform(np.transpose(pos))[:, 0])
        yx_sort = np.stack([pos[i][idx_sort] for i in range(2)], axis=1)
        yx_sort = [tuple(yx_i) for yx_i in yx_sort]
        d_sort = O_iz[pos][idx_sort]
        map_yxd = dict(zip(yx_sort, d_sort))

        # trace until all yx's are exhausted
        yx_traces = []
        d_traces = []
        while len(yx_sort) > 0:
            # trace
            yx_trace_i = []
            d_trace_i = []
            self.dfs2d_from_point(yx_sort[0], map_yxd, yx_trace_i, d_trace_i)
            # record
            yx_traces.append(yx_trace_i)
            d_traces.append(d_trace_i)
            # remove traced yx's
            for yx in yx_trace_i:
                yx_sort.remove(yx)
        
        return yx_traces, d_traces

    def dfs3d(self):
        """ trace 3d binary image self.B
        :return: traces
            traces: [(iz_1, yx_traces_1, d_traces_1),...]
        """
        # flatten array
        traces = []
        for iz in range(self.nz):
            yx_iz, d_iz = self.dfs2d(iz)
            traces.append((iz, yx_iz, d_iz))
        return traces

    #=========================
    # sort coordinates
    #=========================
    
    def sort_coord(self):
        """ sort voxels of image by bfs
        :return: zyx
            zyx: 2d np.ndarray, [[z1,y1,x1],...]
        """
        zyx_arr = []
        # sort for each slice
        for iz in range(self.B.shape[0]):
            # get yx by bfs, directly concat
            yx_traces, _ = self.bfs2d(iz)
            yx_iz = np.concatenate(yx_traces, axis=0)
            # prepend z
            z_iz = iz*np.ones((len(yx_iz), 1), dtype=np.int_)
            zyx_iz = np.concatenate([z_iz, yx_iz], axis=1)
            zyx_arr.append(zyx_iz)
        # concat all
        zyx = np.concatenate(zyx_arr, axis=0)
        return zyx
