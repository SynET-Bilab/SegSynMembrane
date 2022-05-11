import collections
import numpy as np
import sklearn.decomposition
from etsynseg import pcdutils

__all__ = [
    "Tracing"
]

class Tracing:
    """ Tracing connected segments in a stack of 2d images.

    Examples:
        tr = etsynseg.tracing.Tracing(B, O, n_next=2, max_size=10)
        yx_trs, d_trs = tr.bfs2d(iz)
        yx_trs, d_trs = tr.bfs2d(iz)
        zyx = tr.sort_points(method="bfs")
    """
    def __init__(self, B, O, n_next=2, max_size=np.inf):
        """ Initialization.

        Args:
            B, O (np.ndarray): Input binary image and orientation, with shape=(nz,ny,nx).
            n_next (int): The number of candidate pixels to consider for the next visit. Ranges from 1 (next along O) to 3 (plus two neighbors).
            max_size (int): The max size of a segment.
        """
        # save variables
        self.B = B
        self.O = np.copy(O)
        self.nz = B.shape[0]
        self.max_size = max_size
        
        # setup direction
        self.pca = None
        self.set_direction()

        # prep for finding next yx
        self.next_yx_tools = self.prep_find_next_yx(n_next)
    
    #=========================
    # auxiliary functions
    #=========================
    
    def set_direction(self):
        """ Set pca for projected yx. Align O to axis pc1.
        """
        # calc pca
        zyx = pcdutils.pixels2points(self.B)
        self.pca = sklearn.decomposition.PCA(n_components=1)
        self.pca.fit(zyx[:, 1:])
        
        # align O with pc axis 1
        axis_y, axis_x = self.pca.components_[0]
        inner_prod = np.cos(self.O)*axis_x + np.sin(self.O)*axis_y
        self.O[(self.B>0)&(inner_prod<0)] += np.pi

    def prep_find_next_yx(self, n_next):
        """ Preparations for finding next point.
        
        Args:
            n_next (int): The number of candidate pixels for the next visit. See __init__().

        Returns:
            {map_dydx, bins} (dict): Bins for assigning angles from [0,2pi) to 16 discrete codes. Map_dydx maps the code to a list of candidate (dy,dx)'s for the next visit.
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

        # keep first n_next in map_dydx 
        n_next = int(np.clip(n_next, 1, 3))
        for k, v in map_dydx.items():
            map_dydx[k] = v[:n_next]

        bins = np.pi*np.arange(0, 16.1, 1)/8
        return dict(map_dydx=map_dydx, bins=bins)

    def find_next_yx(self, d_curr):
        """ Find next yx candidates according to current direction.

        Args:
            d_curr (float): Direction of the current point.

        Returns:
            dydx_candidates (list of 2-tuple): A list of candidate (dy,dx)'s for the next visit.
        """
        # direction: keep in range [0,2pi)
        d_curr = np.mod(d_curr, 2*np.pi)

        # categorize: into bins from
        loc_bin = np.histogram(d_curr, bins=self.next_yx_tools["bins"])[0].argmax()

        # obtain candidates
        dydx_candidates = self.next_yx_tools["map_dydx"][loc_bin]
        return dydx_candidates

    def find_next_direction(self, d_curr, o_next):
        """ Find the direction of the next point, and align its value to be contiguous with the direction of the current point.

        Args:
            d_curr (float): Direction of the current point, which ranges in [0,2pi).
            o_next (float): Orientation of the next point, which ranges in (-pi/2,pi/2).

        Returns:
            d_next (float): Aligned direction of the next point.
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
        """ Trace a segment from the current point and its direction. Breadth-first scan.

        Args:
            yx_start (2-tuple): The current point, (y,x).
            map_yxd (dict): Maps a point to its direction, each item is like (y,x):direction.

        Returns:
            yx_tr (list of 2-tuple): A list of traced points, [(y1,x1),(y2,x2),...].
            d_tr (list of float): A list of direction corresponding to the points, [d1,d2,...].
        """
        yx_tr = []
        d_tr = []

        # BFS
        yx_tovisit = collections.deque()
        yx_tovisit.append(yx_start)
        while yx_tovisit:
            yx_curr = yx_tovisit.popleft()

            # continue: if yx_curr is not in image, or visited
            if (yx_curr not in map_yxd) or (map_yxd[yx_curr] is None):
                continue

            # visit yx_curr, flag with None
            d_curr = map_yxd[yx_curr]
            yx_tr.append(yx_curr)
            d_tr.append(d_curr)
            map_yxd[yx_curr] = None
            
            # break: if max_size is reached
            if len(yx_tr) >= self.max_size:
                break

            # update yx_tovisit
            for dydx in self.find_next_yx(d_curr):
                yx_next = (yx_curr[0]+dydx[0], yx_curr[1]+dydx[1])
                # align next orientation, update dict
                if (yx_next in map_yxd) and (map_yxd[yx_next] is not None):
                    d_next = self.find_next_direction(d_curr, map_yxd[yx_next])
                    map_yxd[yx_next] = d_next
                    yx_tovisit.append(yx_next)

        return yx_tr, d_tr

    def bfs2d(self, iz):
        """ Trace a 2d image slice using breadth-first scan.

        Args:
            iz (int): Index of the slice in z-direction.

        Returns:
            yx_trs (list): A list of traced segments. Each segment in the list reads [(y1,x1),(y2,x2),...].
            d_trs (list): A list of the direction of points in the segments. Each item in the list reads [d1,d2,...].
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
        yx_trs = []
        d_trs = []
        while len(yx_sort) > 0:
            # trace
            yx_tr_i, d_tr_i = self.bfs2d_from_point(yx_sort[0], map_yxd)
            # record
            yx_trs.append(yx_tr_i)
            d_trs.append(d_tr_i)
            # remove traced yx's
            for yx in yx_tr_i:
                yx_sort.remove(yx)
        
        return yx_trs, d_trs

    #=========================
    # depth-first-scan
    #=========================

    def dfs2d_from_point(self, yx_start, map_yxd):
        """ Trace a segment from the current point and its direction. Depth-first scan.

        Args:
            yx_start (2-tuple): The current point, (y,x).
            map_yxd (dict): Maps a point to its direction, each item is like (y,x):direction.

        Returns:
            yx_tr (list of 2-tuple): A list of traced points, [(y1,x1),(y2,x2),...].
            d_tr (list of float): A list of direction corresponding to the points, [d1,d2,...].
        """
        yx_tr = []
        d_tr = []

        # DFS
        yx_tovisit = collections.deque()
        yx_tovisit.append(yx_start)
        while yx_tovisit:
            yx_curr = yx_tovisit.pop()

            # continue: if yx_curr is not in image, or visited
            if (yx_curr not in map_yxd) or (map_yxd[yx_curr] is None):
                continue

            # visit yx_curr, flag with None
            d_curr = map_yxd[yx_curr]
            yx_tr.append(yx_curr)
            d_tr.append(d_curr)
            map_yxd[yx_curr] = None

            # break: if max_size is reached
            if len(yx_tr) >= self.max_size:
                break

            # update yx_tovisit
            for dydx in self.find_next_yx(d_curr):
                yx_next = (yx_curr[0]+dydx[0], yx_curr[1]+dydx[1])
                # align next orientation, update dict
                if (yx_next in map_yxd) and (map_yxd[yx_next] is not None):
                    d_next = self.find_next_direction(d_curr, map_yxd[yx_next])
                    map_yxd[yx_next] = d_next
                    yx_tovisit.append(yx_next)
                    break

        return yx_tr, d_tr

    def dfs2d(self, iz):
        """ Trace a 2d image slice using depth-first scan.

        Args:
            iz (int): Index of the slice in z-direction.

        Returns:
            yx_trs (list): A list of traced segments. Each segment in the list reads [(y1,x1),(y2,x2),...].
            d_trs (list): A list of the direction of points in the segments. Each item in the list reads [d1,d2,...].
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
        yx_trs = []
        d_trs = []
        while len(yx_sort) > 0:
            # trace
            # yx_tr_i = []
            # d_tr_i = []
            # self.dfs2d_from_point(yx_sort[0], map_yxd, yx_tr_i, d_tr_i)
            yx_tr_i, d_tr_i = self.dfs2d_from_point(yx_sort[0], map_yxd)
            # record
            yx_trs.append(yx_tr_i)
            d_trs.append(d_tr_i)
            # remove traced yx's
            for yx in yx_tr_i:
                yx_sort.remove(yx)
        
        return yx_trs, d_trs

    #=========================
    # sorting points
    #=========================
    
    def sort_points(self, method="bfs"):
        """ Sorting pixels of the image stack by BFS.

        Args:
            method (str): 'bfs' or 'dfs'.

        Returns:
            zyx (np.ndarray): Array of sorted points. Shape=(npts,3), [[z1,y1,x1],...].
        """
        # setup method
        if method == "bfs":
            func_trace = self.bfs2d
        elif method == "dfs":
            func_trace = self.dfs2d
        else:
            raise ValueError("Method should be bfs or dfs.")

        zyx_arr = []
        # sort for each slice
        for iz in range(self.B.shape[0]):
            # get yx, directly concat
            yx_trs, _ = func_trace(iz)
            yx_iz = np.concatenate(yx_trs, axis=0)
            # prepend z
            z_iz = iz*np.ones((len(yx_iz), 1), dtype=np.int_)
            zyx_iz = np.concatenate([z_iz, yx_iz], axis=1)
            zyx_arr.append(zyx_iz)
        # concat all
        zyx = np.concatenate(zyx_arr, axis=0)
        return zyx
