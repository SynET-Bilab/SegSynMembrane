import numpy as np
import sklearn.decomposition


class Grid:
    """ Assigning points into grids
    Usage:
        grid = Grid(zyx, n_vxy, n_uz, nz_eachu)
        grid.uv_size, grid.uv_zyx
    Attributes:
        uv_size: uv_size[iu][iv] is the number of elements in the grid
        uv_zyx: uv_zyx[iu][iv] is the list of [z,y,x] of points in the grid
    """
    def __init__(self, zyx, n_vxy, n_uz, nz_eachu, shrink_sidegrid):
        """ init and generate grids
        :param zyx: points, shape=(npts, 3)
        :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
        :param nz_eachu: number of z-direction slices contained in each grid
        :param shrink_sidegrid: grids close to the side in xy are shrinked to this ratio
        """
        # record of inputs
        self.zyx = zyx
        self.zs = np.unique(self.zyx[:, 0])
        self.yxs = {z: self.zyx[self.zyx[:, 0]==z][:, 1:] for z in self.zs}
        self.nz = len(self.zs)
        self.nz_eachu = nz_eachu
        self.shrink_sidegrid = shrink_sidegrid

        # info of each bin[iu]: indexes of z, coordinates
        self.n_uz = min(n_uz, int(self.nz/self.nz_eachu))
        self.ubin_iz = self.get_ubin_iz()
        self.ubin_zyx = self.get_ubin_zyx()

        # info of each grid[iu][iv]: size, coordinates
        npts_iu = [len(self.ubin_zyx[iu][i])
            for i in range(self.nz_eachu)
            for iu in range(self.n_uz)
        ]
        self.n_vxy = min(n_vxy, np.min(npts_iu))
        self.uv_size, self.uv_zyx = self.get_grid_zyx()

    def get_ubin_iz(self):
        """ for each bin in u(z)-direction, get z-indexes
        :return: ubin_iz
            ubin_iz: shape=(n_uz, nz_eachu), ubin_iz[iu]=(z1,z2,z3,...)
        """
        # split iz into n_uz parts
        iz_split = np.array_split(
            self.zs[:-self.nz_eachu], self.n_uz-1
        )
        # last part = last nz_eachu elements
        iz_split.append(self.zs[-self.nz_eachu:])

        # take the first nz_eachu elements from each part
        ubin_iz = tuple(
            tuple(izs[:self.nz_eachu]) for izs in iz_split
        )
        return ubin_iz

    def get_ubin_zyx(self):
        """ for each bin and slice in u(z)-direction, get coordinates sorted by PC
        :return: ubin_zyx
            ubin_zyx: shape=(n_uz, nz_eachu, n_points, 3), ubin_zyx[iu][i]=((z1,y1,x1),...)
        """
        # pca of all points
        pca = sklearn.decomposition.PCA()
        pca.fit(self.zyx[:, 1:])

        # collect zyx in each u-bin
        ubin_zyx = []
        for iu in range(self.n_uz):
            zyx_iu = []
            # sort zyx for each iz
            for iz in self.ubin_iz[iu]:
                # project yx to pc
                pc1 = pca.transform(self.yxs[iz])[:, 0]
                # sort by pc1, get zyx
                idx_sorted = np.argsort(pc1)
                zyx_iz = tuple(
                    tuple([iz, *self.yxs[iz][i]]) for i in idx_sorted
                )
                zyx_iu.append(zyx_iz)
            ubin_zyx.append(tuple(zyx_iu))
        ubin_zyx = tuple(ubin_zyx)
        return ubin_zyx

    def get_grid_zyx(self):
        """ for each uv-grid, get size and coordinates
        :return: uv_size, uv_zyx
            uv_size: uv_size[(iu,iv)]=n_uv, number of points in the grid
            uv_zyx: uv_zyx[(iu,iv)]=((z1,y1,x1),...), coordinates in the grid
        """
        # initialize
        uv_zyx = {
            (iu, iv): []
            for iu in range(self.n_uz)
            for iv in range(self.n_vxy)
        }

        # populate each grid
        for iu in range(self.n_uz):
            # distribute points in each z-slice into grids in v-direction
            for zyx_iz in self.ubin_zyx[iu]:
                idx_iz = np.arange(len(zyx_iz))
                # set the number of points on the side
                n_side = max(1, int(len(idx_iz)/self.n_vxy*self.shrink_sidegrid))
                # use array split to deal with uneven splits
                idx_split = ([
                    idx_iz[:n_side]]
                    + np.array_split(idx_iz[n_side:-n_side], self.n_vxy-2)
                    + [idx_iz[-n_side:]
                ])
                for iv in range(self.n_vxy):
                    zyx_iz_iv = [zyx_iz[i] for i in idx_split[iv]]
                    uv_zyx[(iu, iv)].extend(zyx_iz_iv)

        # convert to dict of tuples
        uv_zyx = {k: tuple(v) for k, v in uv_zyx.items()}
        # get size
        uv_size = {k: len(v) for k, v in uv_zyx.items()}
        return uv_size, uv_zyx
