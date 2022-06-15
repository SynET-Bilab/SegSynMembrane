import numpy as np
from etsynseg import pcdutil

class Grid:
    """ Assigning points into grids.
    
    Examples:
        grid = Grid(zyx, guide, shrink_sidegrid=0.2, nz_eachu=1)
        grid.gen_grids_by_len(len_grids=(50, 400))

    Attributes:
        uv_size: uv_size[(iu,iv)] is the number of points in the grid.
        uv_zyx: uv_zyx[(iu,iv)] is the array of point coordinates (in [z,y,x]) in the grid.
    """
    def __init__(self, zyx, guide, shrink_sidegrid=0.25, nz_eachu=1):
        """ Initialization.

        Read points and sort by the guide.
        The number of grids will be set by self.gen_grids_by_num or self.gen_grids_by_len.

        Args:
            zyx (np.ndarray): 3d points, with shape=(npts,3). Each point is [zi,yi,xi].
            guide (np.ndarray): 3d guideline points sorted in each slice, with shape=(npts_guide,3). Each point is [zi,yi,xi].
            nz_eachu (int): The number of z-direction slices contained in each grid.
            shrink_sidegrid (float): Grids close to the sides in xy are shrinked to this ratio.
                A smaller ratio facilitates the coverage of the side in sampling.
        """
        # sort points by the guide
        self.guide = pcdutil.points_deduplicate(guide)
        self.zyx = pcdutil.sort_pts_by_guide(zyx, self.guide)
        # setup basic info
        self.zs = sorted(np.unique(self.zyx[:, 0]))
        self.nz = len(self.zs)
        self.nz_eachu = nz_eachu
        self.shrink_sidegrid = shrink_sidegrid

    def gen_grids_by_num(self, n_uz, n_vxy):
        """ Given the number of grids, then generate grids.

        Constraints:
            n_uz < min number of slices / nz_eachu
            n_vxy < min number of points in all slices * nz_eachu

        Args:
            n_uz, n_vxy (int): The number of sampling grids in v(xy) and u(z) directions.
        
        Returns:
            self (Grid): Self object with set number of grids.
        """
        # set the number of grids
        min_npts_z = np.min([
            np.sum(self.zyx[:, 0]==z) for z in self.zs
        ])
        self.n_vxy = min(n_vxy, int(min_npts_z*self.nz_eachu))
        self.n_uz = min(n_uz, int(self.nz/self.nz_eachu))

        # generate grids
        # info of each bin[iu]: indexes of z, coordinates
        self.ubin_iz = self.gen_ubin_iz()
        self.ubin_zyx = self.gen_ubin_zyx()

        # info of each grid[iu][iv]: size, coordinates
        npts_iu = [
            len(self.ubin_zyx[iu][i])
            for i in range(self.nz_eachu)
            for iu in range(self.n_uz)
        ]
        self.n_vxy = min(self.n_vxy, np.min(npts_iu))
        self.uv_size, self.uv_zyx = self.gen_grid_zyx()
        return self

    def gen_grids_by_len(self, len_grids, ngrids_min=(3, 3)):
        """ Set the number of grids by providing target lengths.

        Args:
            len_grids (2-tuple of float): The length of grids in u(z),v(xy) directions, (len_uz,len_vxy) in units of pixels.
            ngrids_min (2-tuple of int): The minimum number of grids in u(z),v(xy) directions, (n_uz,n_vxy).
        
        Returns:
            self (Grid): Self object with set number of grids.
        """
        len_uz, len_vxy = len_grids
        n_uz_min, n_vxy_min = ngrids_min

        # grids in u(z) direction
        n_uz = int(np.round(np.ptp(self.zs)/len_uz))+1
        n_uz = max(n_uz_min, n_uz)

        # grids in v(xy) direction
        # avg length of guide in xy direction
        len_guide = 0
        for z in self.zs:
            len_guide += pcdutil.wireframe_length(
                self.guide[self.guide[:, 0] == z][::2]
            )
        len_guide /= len(self.zs)
        # (n_vxy-2)*len_vxy + 2*shrink_sidegrid*len_vxy = len_guide
        n_vxy = len_guide/len_vxy + 2*(1-self.shrink_sidegrid)
        n_vxy = int(np.round(n_vxy)) + 1
        n_vxy = max(n_vxy_min, n_vxy)

        # generate grids by number
        self.gen_grids_by_num(n_uz=n_uz, n_vxy=n_vxy)
        return self

    def get_config(self, downscale=1):
        """ Get grid config as a dict.
        
        Downscaling can reduce the degrees of freedom, which makes MOO easier.
        The number of grids remain unchanged.
        The coordinates are downscaled, made int, deduplicated.

        Args:
            downscale (float): Factor for downscaling.
        
        Returns:
            config (dict): Dict for configurations.
                {downscale,zyx,count_zyx,n_uz,n_vxy,uv_zyx,uv_size}
                downscale: regularized downscale factor, so that downscaled z in different grids do not overlap
                zyx: downscaled rounded points
                count_zyx: the count of each downscaled point before deduplicating
        """
        config = {}
        
        # setup downscaling factor
        # so that downscaling would not cause z in different grids to overlap
        dz_arr = []
        for iu in range(1, self.n_uz):
            dz = self.uv_zyx[(iu, 0)][:, 0].min() - self.uv_zyx[(iu-1, 0)][:, 0].max()
            dz_arr.append(dz)
        dz_min = np.min(dz_arr)
        downscale = max(downscale, 1)  # downscale>=1
        downscale = min(downscale, dz_min)  # dz_min/downscale>=1
        config["downscale"] = downscale

        # downscale coordinates, count for each downscaled point
        zyx_down = self.zyx/downscale
        config["zyx"], config["count_zyx"] = pcdutil.points_deduplicate_count(zyx_down)
        
        # downscale grids
        config["n_uz"] = self.n_uz
        config["n_vxy"] = self.n_vxy
        config["uv_zyx"] = {}
        config["uv_size"] = {}
        for iu in range(self.n_uz):
            for iv in range(self.n_vxy):
                zyx_i = pcdutil.points_deduplicate(self.uv_zyx[(iu, iv)]/downscale)
                config["uv_zyx"][(iu, iv)] = zyx_i
                config["uv_size"][(iu, iv)] = len(zyx_i)
        return config

    def gen_ubin_iz(self):
        """ Get z-indexes for each bin in u(z)-direction.

        Returns:
            ubin_iz (tuple): shape=(n_uz,nz_eachu), ubin_iz[iu]=(z1,z2,z3,...).
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

    def gen_ubin_zyx(self):
        """ Get points for each bin and slice in u(z)-direction.

        Returns:
            ubin_zyx (tuple): Points in each bin and slice, shape=(n_uz,nz_eachu,n_points,3).
                ubin_zyx[iu][i]=[[z0,y0,x0],...].
        """
        # collect zyx in each u-bin
        ubin_zyx = []
        for iu in range(self.n_uz):
            zyx_iu = []
            # append zyx in each iz
            for iz in self.ubin_iz[iu]:
                zyx_iu.append(self.zyx[self.zyx[:, 0] == iz])
            ubin_zyx.append(zyx_iu)
        return ubin_zyx

    def gen_grid_zyx(self):
        """ For each uv-grid, get size and coordinates.

        Returns:
            uv_size (dict): The number of points in grid (iu,iv), uv_size[(iu,iv)]=n_uv.
            uv_zyx (dict): Coordinates in the grid (iu,iv), uv_zyx[(iu,iv)]=[[z0,y0,x0],...].
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
                n_side = int(len(idx_iz)/self.n_vxy*self.shrink_sidegrid)
                n_side = max(1, n_side)
                # use array split to deal with uneven splits
                idx_split = (
                    [idx_iz[:n_side]]
                    + np.array_split(idx_iz[n_side:-n_side], self.n_vxy-2)
                    + [idx_iz[-n_side:]]
                )
                for iv in range(self.n_vxy):
                    zyx_iz_iv = [zyx_iz[i] for i in idx_split[iv]]
                    uv_zyx[(iu, iv)].append(zyx_iz_iv)

        # construct dicts
        uv_zyx = {
            k: np.concatenate(v, axis=0)
            for k, v in uv_zyx.items()
        }
        uv_size = {k: len(v) for k, v in uv_zyx.items()}
        return uv_size, uv_zyx
