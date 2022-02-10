import numpy as np
import sklearn.decomposition

from etsynseg.utils import mask_to_coord


class Grid:
    """ Assigning points into grids
    Usage:
        grid = Grid(B, n_vxy, n_uz, nz_eachu)
        grid.uv_size, grid.uv_zyx
    Attributes:
        uv_size: uv_size[iu][iv] is the number of elements in the grid
        uv_zyx: uv_zyx[iu][iv] is the list of [z,y,x] of points in the grid
    """
    def __init__(self, B, n_vxy, n_uz, nz_eachu):
        """ init and generate grids
        :param B: binary image, shape=(nz,ny,nx)
        :param n_vxy, n_uz: number of sampling grids in v(xy) and u(z) directions
        :param nz_eachu: number of z-direction slices contained in each grid
        """
        # record of inputs
        self.B = B
        self.nz = self.B.shape[0]
        self.nz_eachu = nz_eachu

        # nv, nu: try to avoid empty grids
        # nv: <= min number of pixels among z's
        max_nv = np.sum(self.B, axis=(1, 2)).min()
        self.n_vxy = min(n_vxy, max_nv)
        # nu: <= nz/nz_eachu
        max_nu = int(self.nz/self.nz_eachu)
        self.n_uz = min(n_uz, max_nu)

        # info of each bin[iu]: indexes of z, coordinates
        self.ubin_iz = self.get_ubin_iz()
        self.ubin_zyx = self.get_ubin_zyx()

        # info of each grid[iu][iv]: size, coordinates
        self.uv_size, self.uv_zyx = self.get_grid_zyx()

    def get_ubin_iz(self):
        """ for each bin in u(z)-direction, get z-indexes
        :return: ubin_iz
            ubin_iz: shape=(n_uz, nz_eachu), ubin_iz[iu]=(z1,z2,z3,...)
        """
        # interval between u-bins
        # used np.array_split to deal with uneven intervals
        interval_size = (self.nz - self.n_uz*self.nz_eachu)
        interval_arr = np.array_split(np.ones(interval_size), self.n_uz-1)

        # z-indexes for each u-bin
        ubin_iz = []
        iz = 0
        for i in range(self.n_uz-1):
            # take samples
            ubin_iz.append(tuple(range(iz, iz+self.nz_eachu)))
            # skip interval
            iz += self.nz_eachu + len(interval_arr[i])
        # take the last bin
        ubin_iz.append(tuple(range(self.nz-self.nz_eachu, self.nz)))

        ubin_iz = tuple(ubin_iz)
        return ubin_iz

    def get_ubin_zyx(self):
        """ for each bin and slice in u(z)-direction, get coordinates sorted by PC
        :return: ubin_zyx
            ubin_zyx: shape=(n_uz, nz_eachu, n_points, 3), ubin_zyx[iu][i]=((z1,y1,x1),...)
        """
        # pca of all points
        zyx_all = mask_to_coord(self.B)
        pca = sklearn.decomposition.PCA()
        pca.fit(zyx_all[:, 1:])

        # collect zyx in each u-bin
        ubin_zyx = []
        for iu in range(self.n_uz):
            zyx_iu = []
            # sort zyx for each iz
            for iz in self.ubin_iz[iu]:
                # project yx to pc
                yx_iz = mask_to_coord(self.B[iz])
                pc1 = pca.transform(yx_iz)[:, 0]
                # sort by pc1, get zyx
                idx_sorted = np.argsort(pc1)
                zyx_iz = tuple(
                    tuple([iz, *yx_iz[i]]) for i in idx_sorted
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
                # use array split to deal with uneven splits
                idx_split = np.array_split(idx_iz, self.n_vxy)
                for iv in range(self.n_vxy):
                    zyx_iz_iv = [zyx_iz[i] for i in idx_split[iv]]
                    uv_zyx[(iu, iv)].extend(zyx_iz_iv)

        # convert to dict of tuples
        uv_zyx = {k: tuple(v) for k, v in uv_zyx.items()}
        # get size
        uv_size = {k: len(v) for k, v in uv_zyx.items()}
        return uv_size, uv_zyx
