""" 
"""

import numpy as np
from etsynseg import io, utils, trace
from etsynseg import hessian, dtvoting, nonmaxsup
from etsynseg import evomsac, matching

__all__ = [
    "SegBase", "SegSteps"
]
class SegBase:
    """ workflow for segmentation
    attribute formats:
        binary images: save coordinates (utils.mask_to_coord, self.coord_to_binary)
        sparse images (O): save as sparse (utils.sparsify3d, utils.densify3d)
        MOOPop: save state (MOOPop().dump_state, MOOPop(state=state))
    """
    def __init__(self):
        self.steps = {}

    def view_status(self):
        """ view status (finished, timing) of each step
        """
        status = {
            k: {"finished": v["finished"], "timing": v["timing"]}
            for k, v in self.steps.items()
        }
        return status
    
    def save_steps(self, filename):
        """ save steps to npz
        :param filename: filename to save to
        """
        np.savez_compressed(filename, **self.steps)
    
    def load_steps(self, filename):
        """ load steps from npz
        :param filename: filename to load from
        """
        steps = np.load(filename, allow_pickle=True)
        self.steps = {key: steps[key].item() for key in steps.keys()}

    def check_steps(self, steps_prev, raise_error=False):
        """ raise error if any prerequisite steps is not finished
        :param steps_prev: array of names of prerequisite steps
        :param raise_error: if raise error when prerequisites are not met
        """
        satisfied = True
        for step in steps_prev:
            if not self.steps[step]["finished"]:
                if raise_error:
                    raise RuntimeError(f"unsatisfied prerequisite step: {step}")
                satisfied = False
        return satisfied


#=========================
# auxiliary
#=========================

class SegSteps:
    @staticmethod
    def set_ngrids(B, voxel_size_nm, grid_z_nm, grid_xy_nm):
        """ setting number of grids for evomsac and fitting
        :param B: binary image
        :param voxel_size_nm: voxel spacing in nm
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :return: n_uz, n_vxy
        """
        # auxiliary
        def set_one(span_vx, grid_nm, n_min):
            span_nm = span_vx * voxel_size_nm
            n_grid = int(np.round(span_nm/grid_nm)+1)
            return max(n_min, n_grid)

        # grids in z
        nz = B.shape[0]
        n_uz = set_one(nz, grid_z_nm, 3)

        # grids in xy
        dydx = utils.spans_xy(B)
        l_xy = np.median(np.linalg.norm(dydx, axis=1))
        n_vxy = set_one(l_xy, grid_xy_nm, 3)
        
        return n_uz, n_vxy
    
    # @staticmethod
    # def sort_coord(zyx):
    #     """ sort coordinates by pc
    #     :param zyx: zyx
    #     :return: zyx_sorted
    #     """
    #     # fit pca
    #     pca = sklearn.decomposition.PCA(n_components=1)
    #     pca.fit(zyx[:, 1:])

    #     # sort for each z
    #     iz_arr = sorted(np.unique(zyx[:, 0]))
    #     zyx_sorted_arr = []
    #     for iz in iz_arr:
    #         zyx_iz = zyx[zyx[:, 0] == iz]
    #         pc1_iz = pca.transform(zyx_iz[:, 1:])[:, 0]
    #         idx_sort = np.argsort(pc1_iz)
    #         zyx_sorted_arr.append(zyx_iz[idx_sort])
        
    #     zyx_sorted = np.concatenate(zyx_sorted_arr)
    #     return zyx_sorted

    @staticmethod
    def read_tomo(tomo_file, model_file,
            voxel_size_nm=None, d_mem_nm=5, obj_bound=1
        ):
        """ load and clip tomo and model
        :param tomo_file, model_file: filename of tomo, model
        :param voxel_size_nm: if None then read from tomo_file
        :param d_mem_nm: lengthscale of membrane thickness
        :param obj_bound: obj label for boundary
        :return: results
            results: {tomo_file,model_file,obj_bound,d_mem_nm,
                I,shape,voxel_size_nm,model,clip_range,zyx_shift,zyx_bound,contour_bound,contour_len_bound,d_mem}
        """
        # read model
        model = io.read_model(model_file)
        
        # set the range of clipping
        # use np.floor/ceil -> int to ensure integers
        model_bound = model[model["object"] == obj_bound]
        clip_range = {
            i: (int(np.floor(model_bound[i].min())),
                int(np.ceil(model_bound[i].max())))
            for i in ["x", "y", "z"]
        }
        zyx_shift = np.array([clip_range[i][0] for i in ['z', 'y', 'x']])

        # read tomo, clip to bound
        I, voxel_size_A = io.read_clip_tomo(
            tomo_file, clip_range
        )
        I = np.asarray(I)

        # voxel_size in nm
        if voxel_size_nm is None:
            # assumed voxel_size_A is the same for x,y,z
            voxel_size_nm = voxel_size_A.tolist()[0] / 10

        # shift model coordinates
        for i in ["x", "y", "z"]:
            model[i] -= clip_range[i][0]

        # generate mask for bound (after clipping)
        mask_bound = io.model_to_mask(
            model=model[model["object"] == obj_bound],
            yx_shape=I[0].shape
        )

        # calculate bound contours and lengths
        contour_bound = utils.mask_to_contour(mask_bound)
        contour_bound = np.round(contour_bound).astype(np.int_)
        contour_len_bound = []
        for iz in range(I.shape[0]):
            yx_iz = contour_bound[contour_bound[:, 0] == iz][:, 1:]
            # step=2 to avoid zigzags
            contour_len_iz = np.sum(np.linalg.norm(
                np.diff(yx_iz[::2], axis=0), axis=1))
            contour_len_bound.append(contour_len_iz)
        contour_len_bound = np.array(contour_len_bound)

        # save parameters and results
        results = dict(
            # parameters
            tomo_file=tomo_file,
            model_file=model_file,
            obj_bound=obj_bound,
            d_mem_nm=d_mem_nm,
            # results
            I=I,
            shape=I.shape,
            voxel_size_nm=voxel_size_nm,
            model=model,
            clip_range=clip_range,
            zyx_shift=zyx_shift,
            zyx_bound=utils.mask_to_coord(mask_bound),
            contour_bound=contour_bound,
            contour_len_bound=contour_len_bound,
            d_mem=d_mem_nm/voxel_size_nm,
        )
        return results

    @staticmethod
    def detect(I, mask_bound, contour_len_bound,
            sigma_hessian, sigma_tv, sigma_supp,
            dO_threshold=np.pi/4, xyfilter=2.5, dzfilter=1
        ):
        """ detect membrane features
        :param sigma_<hessian,tv,supp>: sigma in voxel for hessian, tv, normal suppression
        :param xyfilter: for each xy plane, filter out pixels with Ssupp below quantile threshold; the threshold = 1-xyfilter*fraction_mems, fraction_mems is estimated by the ratio between contour length of boundary and the number of points. the smaller xyfilter, more will be filtered out.
        :param dzfilter: a component will be filtered out if its z-range < dzfilter
        :return: results
            results: {qfilter,zyx_supp,zyx,Oz}
        """
        # negate, hessian
        Ineg = utils.negate_image(I)
        S, O = hessian.features3d(Ineg, sigma=sigma_hessian)
        
        # tv, nms
        Stv, Otv = dtvoting.stick3d(
            S*mask_bound, O*mask_bound, sigma=sigma_tv
        )
        Btv = mask_bound*nonmaxsup.nms3d(Stv, Otv)
        
        # refine: orientation, nms, orientation changes
        _, Oref = hessian.features3d(Btv, sigma=sigma_hessian)
        Bref = nonmaxsup.nms3d(Stv*Btv, Oref*Btv)
        
        # normal suppression
        Bsupp, Ssupp = dtvoting.suppress_by_orient(
            Bref, Oref*Bref,
            sigma=sigma_supp,
            dO_threshold=dO_threshold
        )
        Ssupp = Ssupp * Bsupp

        # filter in xy: small Ssupp
        Bfilt_xy = np.zeros(Bsupp.shape, dtype=int)
        for iz in range(Bsupp.shape[0]):
            # estimating the fraction of membranes by no. of pts
            npts_bound = contour_len_bound[iz]
            npts_ref = np.sum(Bref[iz])
            fraction_mems = npts_bound/npts_ref
            
            # set thresh according to the fraction
            qthresh = 1 - np.clip(xyfilter*fraction_mems, 0, 1)
            
            # filter out pixels with small Ssupp
            ssupp = Ssupp[iz][Bsupp[iz].astype(bool)]
            sthresh = np.quantile(ssupp, qthresh)
            
            # assign filtered results
            Bfilt_xy[iz][Ssupp[iz]>sthresh] = 1

        # filter in 3d: z-span
        Bfilt_dz = utils.filter_connected_dz(
            Bfilt_xy, dzfilter=dzfilter, connectivity=3)

        # settle detected
        Bdetect = Bfilt_dz
        Odetect = Oref*Bdetect

        # save parameters and results
        results = dict(
            # parameters
            sigma_tv=sigma_tv,
            sigma_supp=sigma_supp,
            xyfilter=xyfilter,
            dzfilter=dzfilter,
            # results
            zyx_supp=trace.Trace(Bsupp, Odetect*Bsupp).sort_coord(),
            zyx=trace.Trace(Bdetect, Odetect*Bdetect).sort_coord(),
            Oz=utils.sparsify3d(Odetect)
        )
        return results

    @staticmethod
    def evomsac(B, voxel_size_nm, grid_z_nm, grid_xy_nm,
            pop_size, max_iter, tol
        ):
        """ evomsac for one divided part
        :param B: 3d binary image
        :param voxel_size_nm: voxel spacing in nm
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :param pop_size: size of population
        :param tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
        :param max_iter: max number of generations
        """
        # setup grid and mootools
        n_uz, n_vxy = SegSteps.set_ngrids(B, voxel_size_nm, grid_z_nm, grid_xy_nm)
        mtools = evomsac.MOOTools(
            B, n_vxy=n_vxy, n_uz=n_uz, nz_eachu=1, r_thresh=1
        )

        # setup pop, evolve
        mpop = evomsac.MOOPop(mtools, pop_size=pop_size)
        mpop.init_pop()
        mpop.evolve(max_iter=max_iter, tol=tol)
        return mpop

    @staticmethod
    def match(B, O, mpop, sigma_tv, sigma_hessian, sigma_extend, mask_bound=None):
        """ match for one divided part
        :param B, O: 3d binary image, orientation
        :param mpop: MOOPop from evomsac_one
        :param sigma_tv: sigma for tv enhancement of B, so that lines gets more connected
        :param sigma_<hessian,extend>: sigmas for hessian and tv extension of evomsac'ed image
        :param mask_bound: 3d binary mask for bounding polygon
        :return: Bsmooth, zyx_sorted
            Bsmooth: resulting binary image from matching
            zyx_sorted: coordinates sorted by tracing
        """
        # fit surface
        indiv = mpop.select_by_hypervolume(mpop.log_front[-1])
        pts_net = mpop.mootools.get_coord_net(indiv)
        nu_eval = np.max(utils.wireframe_lengths(pts_net, axis=0))
        nv_eval = np.max(utils.wireframe_lengths(pts_net, axis=1))
        Bfit, _ = mpop.mootools.fit_surface_eval(
            indiv,
            u_eval=np.linspace(0, 1, 2*int(nu_eval)),
            v_eval=np.linspace(0, 1, 2*int(nv_eval))
        )

        # tv
        Stv, Otv = dtvoting.stick3d(B, O, sigma=sigma_tv)
        Btv = nonmaxsup.nms3d(Stv, Otv)

        # matching
        Bmatch = matching.match_spatial_orient(
            Btv, Otv*Btv, Bfit,
            sigma_hessian=sigma_hessian,
            sigma_tv=sigma_extend
        )
        Bmatch = next(iter(utils.extract_connected(Bmatch)))[1]

        # # smoothing
        # _, Osmooth = hessian.features3d(Bmatch, sigma_hessian)
        # Bsmooth = nonmaxsup.nms3d(Bmatch, Osmooth*Bmatch)
        # Bsmooth = next(iter(utils.extract_connected(Bsmooth)))[1]

        # bounding
        if mask_bound is not None:
            Bmatch = Bmatch * mask_bound

        # ordering
        zyx_sorted = trace.Trace(Bmatch, O*Bmatch).sort_coord()
        
        return Bmatch, zyx_sorted

    @staticmethod
    def surf_normal(zyx, zyx_ref, shape, d_mem):
        """ calculate surface normal for one divided part
        :param zyx: coordinates
        :param zyx_ref: reference zyx
        :param shape: (nz,ny,nx) of 3d image
        :param d_mem: membrane thickness in voxel
        :return: normal
            normal: each=(nx,ny,nz)
        """
        B = utils.coord_to_mask(zyx, shape)
        pos = tuple(zyx.T)
        _, normal = hessian.surface_normal(
            B, sigma=d_mem, zyx_ref=zyx_ref, pos=pos
        )
        return normal

    @staticmethod
    def surf_fit(B, voxel_size_nm, grid_z_nm, grid_xy_nm):
        """ surface fitting for one divided part
        :param B: 3d binary image
        :param voxel_size_nm: voxel spacing in nm
        :param grid_z_nm, grid_xy_nm: grid spacing in z, xy
        :return: Bfit, zyx_sorted
        """
        # setup grids, mtools
        n_uz, n_vxy = SegSteps.set_ngrids(B, voxel_size_nm, grid_z_nm, grid_xy_nm)
        mtools = evomsac.MOOTools(
            B, n_uz=n_uz, n_vxy=n_vxy, nz_eachu=1, r_thresh=1
        )

        # generate indiv
        indiv = mtools.indiv_middle()
        
        # fit
        pts_net = mtools.get_coord_net(indiv)
        nu_eval = np.max(utils.wireframe_lengths(pts_net, axis=0))
        nv_eval = np.max(utils.wireframe_lengths(pts_net, axis=1))
        Bfit, _ = mtools.fit_surface_eval(
            indiv,
            u_eval=np.linspace(0, 1, 5*int(nu_eval)),
            v_eval=np.linspace(0, 1, 5*int(nv_eval))
        )

        # ordering
        _, Ofit = hessian.features3d(Bfit, 1)
        zyx_sorted = trace.Trace(Bfit, Ofit*Bfit).sort_coord()

        return Bfit, zyx_sorted
