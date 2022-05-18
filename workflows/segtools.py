""" 
"""

import numpy as np
import multiprocessing.dummy

from etsynseg import matching, meshrefine, nonmaxsup, tracing
from etsynseg import imgutil, pcdutil, modutil, io
from etsynseg import features, dtvoting
from etsynseg import moosac

__all__ = [
    "SegBase", "SegSteps"
]

class SegBase:
    """ workflow for segmentation
    attribute formats:
        binary images: save coordinates (utils.pixels2points, self.coord_to_binary)
        sparse images (O): save as sparse (utils.sparsify3d, utils.densify3d)
        MOOPop: save state (MOOPop().save_state, MOOPop(state=state))
    """
    def __init__(self):
        self.steps = {}

    def view_status(self):
        """ view status (finished, process_time) of each step
        """
        status = {
            k: {"finished": v["finished"], "process_time": v["timing"]}
            for k, v in self.steps.items()
        }
        return status
    
    def save_steps(self, filename):
        """ save steps to npz
            filename: filename to save to
        """
        np.savez_compressed(filename, **self.steps)
    
    def load_steps(self, filename):
        """ load steps from npz
            filename: filename to load from
        """
        steps = np.load(filename, allow_pickle=True)
        self.steps = {key: steps[key].item() for key in steps.keys()}
        return self

    def check_steps(self, steps_prev, raise_error=False):
        """ raise error if any prerequisite steps is not finished
            steps_prev: array of names of prerequisite steps
            raise_error: if raise error when prerequisites are not met
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
    # @staticmethod
    # def sort_points(zyx):
    #     """ sort coordinates by pc
    #         zyx: zyx
    #     Returns: zyx_sorted
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
    def read_tomo_model(tomo_file, model_file, extend_nm, pixel_nm=None, interp_degree=2):
        """ Read tomo and model.

        Read model: object 1 for guide lines, object 2 for reference point.
        Generate region mask surrounding the guide lines.
        Decide clip range from the region mask.
        Read tomo, clip.

        Args:
            tomo_file (str): Filename of tomo mrc.
            model_file (str): Filename of imod model.
            extend_nm (float): Extend from guide lines by this width (in nm) to get the bound.
            pixel_nm (float): Pixel size in nm. If None then read from tomo.
            interp_degree (int): Degree of bspline interpolation of the model.
                2 for most cases.
                1 for finely-drawn model to preserve the original contours.
    
        Returns:
            results (dict): Tomo, model, and relevant info. See below for fields in the dict.
                tomo_file, model_file: filenames
                I: clipped tomo
                shape: I.shape
                pixel_nm: pixel size in nm
                model: model DataFrame, in the original coordinates
                clip_low: [z,y,x] at the lower corner for clipping
                mask_bound, guide, mask_plus, mask_minus: zyx-points in the masks generated from the guide lines
                normal_ref: reference point inside for normal orientation
                }
        """
        # read tomo
        I, pixel_A = io.read_tomo(tomo_file, mode="mmap")
        # get pixel size in nm
        if pixel_nm is None:
            pixel_nm = pixel_A / 10

        # read model
        model = io.read_model(model_file)
        # object: guide lines
        if 1 in model["object"].values:
            model_guide = model[model["object"]==1][['z','y','x']].values
        else:
            raise ValueError(f"The object for guide lines is not found in the model file.")
        # object: normal ref point
        if 2 in model["object"].values:
            normal_ref = model[model["object"]==2][['z','y','x']].values[0]
        else:
            normal_ref = None

        # generate mask from guide line
        guide, mask_plus, mask_minus, normal_ref = modutil.mask_from_model(
            model_guide,
            width=extend_nm/pixel_nm,
            normal_ref=normal_ref,
            interp_degree=interp_degree,
            cut_end=True
        )
        # combine all parts of the mask
        mask_bound = np.concatenate([guide, mask_plus, mask_minus], axis=0)
        mask_bound = np.unique(mask_bound, axis=0)
        
        # get clip range and raw shape from the mask
        clip_low, _, shape = pcdutil.points_range(mask_bound, margin=0)
        
        # clip coordinates
        guide -= clip_low
        mask_plus -= clip_low
        mask_minus -= clip_low
        mask_bound -= clip_low
        normal_ref -= clip_low

        # clip tomo
        sub = tuple(slice(ci, ci+si) for ci, si in zip(clip_low, shape))
        I = np.asarray(I[sub])
        shape = I.shape
        
        # save parameters and results
        results = dict(
            # parameters
            tomo_file=tomo_file,
            model_file=model_file,
            # results
            I=I,
            shape=shape,
            pixel_nm=pixel_nm,
            model=model,
            clip_low=clip_low,
            mask_bound=mask_bound,
            normal_ref=normal_ref,
            guide=guide,
            mask_plus=mask_plus,
            mask_minus=mask_minus
        )
        return results

    @staticmethod
    def detect(
            I, mask_bound, contour_len_bound,
            sigma_hessian, sigma_tv, sigma_supp,
            dO_thresh, xyfilter, dzfilter
        ):
        """ detect membrane features
            sigma_<hessian,tv,supp>: sigma in voxel for hessian, tv, normal suppression
            xyfilter: for each xy plane, filter out pixels with Ssupp below quantile threshold; the threshold = 1-xyfilter*fraction_mems, fraction_mems is estimated by the ratio between contour length of boundary and the number of points. the smaller xyfilter, more will be filtered out.
            dzfilter: a component will be filtered out if its z-range < dzfilter
        Returns: zyx_nofilt, zyx, Oz
            zyx_nofilt: points before filtering (normal suppression and other filters)
            zyx: points after filtering
            Oz: orientation sparsified
        """
        # negate, hessian
        Ineg = utils.negate_image(I)
        S, O = features.features3d(Ineg, sigma=sigma_hessian)
        B = mask_bound*nonmaxsup.nms3d(S, O)

        # tv, nms
        if sigma_tv > 0:
            Stv, Otv = dtvoting.stick3d(
                B, O*B, sigma=sigma_tv
            )
            Btv = mask_bound*nonmaxsup.nms3d(Stv, Otv)
        else:
            Stv = S
            Otv = O
            Btv = B
        
        # refine: orientation, nms, orientation changes
        _, Oref = features.features3d(Btv, sigma=sigma_hessian)
        Bref = nonmaxsup.nms3d(Stv*Btv, Oref*Btv)
        
        # normal suppression
        Bsupp, Ssupp = dtvoting.suppress_by_orient(
            Bref, Oref*Bref,
            sigma=sigma_supp,
            dO_thresh=dO_thresh
        )
        Ssupp = Ssupp * Bsupp

        # filter in xy: small Ssupp
        Bfilt_xy = np.zeros(Bsupp.shape, dtype=int)

        def filter_xy_one(iz):
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
            Bfilt_xy[iz][Ssupp[iz] > sthresh] = 1
        
        pool = multiprocessing.dummy.Pool()
        pool.map(filter_xy_one, range(Bsupp.shape[0]))
        pool.close()

        # filter in 3d: z-span
        Bfilt_dz = utils.filter_connected_dz(
            Bfilt_xy, dzfilter=dzfilter, connectivity=3)

        # settle detected
        Bdetect = Bfilt_dz
        Odetect = Oref*Bdetect

        # retuls
        zyx_nofilt = utils.pixels2points(Bref)
        zyx = utils.pixels2points(Bdetect)
        Oz = imgutil.sparsify3d(Odetect)
        return zyx_nofilt, zyx, Oz

    @staticmethod
    def evomsac(
            zyx, pixel_nm, grid_z_nm, grid_xy_nm,
            shrink_sidegrid, fitness_rthresh,
            pop_size, max_iter, tol, factor_eval
        ):
        """ evomsac for one divided part
            zyx: 3d binary image
            pixel_nm: voxel spacing in nm
            grid_z_nm, grid_xy_nm: grid spacing in z, xy
            shrink_sidegrid: grids close to the side in xy are shrinked to this ratio
            fitness_rthresh: distance threshold for fitness evaluation, r_outliers >= fitness_rthresh
            pop_size: size of population
            tol: (tol_value, n_back), terminate if change ratio < tol_value within last n_back steps
            max_iter: max number of generations
            factor_eval: factor for assigning evaluation points
        Returns: zyx_sac, mpopz
            zyx_sac: zyx after evomsac
            mpopz: mpop state, load by mpop=SegSteps.evomsac_mpop_load(mpopz, zyx)
        """
        # setup grids
        zs = np.unique(zyx[:, 0])
        yxs = {z: zyx[zyx[:, 0] == z][:, 1:] for z in zs}
        # grids in z
        n_uz = len(zs)*pixel_nm/grid_z_nm
        n_uz = max(3, int(np.round(n_uz)+1))
        # grids in xy
        l_xy = np.median(
            [np.linalg.norm(np.ptp(yxs[z], axis=0)) for z in zs]
        )
        n_vxy = l_xy*pixel_nm/grid_xy_nm + 2*(1-shrink_sidegrid)
        n_vxy = max(3, int(np.round(n_vxy)+1))

        # setup mootools, moopop
        mtools = moosac.MOOTools(
            zyx, n_vxy=n_vxy, n_uz=n_uz, nz_eachu=1,
            fitness_rthresh=fitness_rthresh,
            shrink_sidegrid=shrink_sidegrid
        )
        mpop = moosac.MOOPop(mtools, pop_size=pop_size)

        # evolve
        mpop.init_pop()
        mpop.evolve(max_iter=max_iter, tol=tol)

        # fit surface
        # indiv = mpop.select_by_hypervolume(mpop.log_front[-1])
        indiv = mpop.log_front[-1][0]
        pts_net = mpop.mootools.get_coords_net(indiv)
        nu_eval = np.max(utils.wireframe_length(pts_net, axis=0))
        nv_eval = np.max(utils.wireframe_length(pts_net, axis=1))
        zyx_sac, _ = mpop.mootools.fit_surface(
            indiv,
            u_eval=np.linspace(0, 1, factor_eval*int(nu_eval)),
            v_eval=np.linspace(0, 1, factor_eval*int(nv_eval))
        )

        # save mpop state
        mpopz = SegSteps.evomsac_mpop_save(mpop, save_zyx=False)
        return zyx_sac, mpopz

    @staticmethod
    def evomsac_mpop_save(mpop, save_zyx=False):
        """ save moopop
            mpop: MOOPop
            save_zyx: whether to save zyx
        Returns: mpopz
            mpopz: state of mpop
        """
        mpopz = mpop.save_state()
        if not save_zyx:
            mpopz["mootools_config"]["zyx"] = None
        return mpopz
    
    @staticmethod
    def evomsac_mpop_load(mpopz, zyx=None):
        """ load moopop
            mpopz: MOOPop state
            zyx: coordinates
        Returns: mpop
        """
        if zyx is not None:
            mpopz["mootools_config"]["zyx"] = zyx
        mpop = moosac.MOOPop(state=mpopz)
        return mpop

    @staticmethod
    def match(
            B, O, Bsac,
            sigma_tv, sigma_hessian, sigma_extend, mask_bound=None
        ):
        """ match for one divided part
            B, O: 3d binary image and orientation from detected
            Bsac: 3d binary image from evomsac
            sigma_tv: sigma for tv enhancement of B, so that lines gets more connected; if 0 then skip tv
            sigma_<hessian,extend>: sigmas for hessian and tv extension of evomsac'ed image
            mask_bound: 3d binary mask for bounding polygon
        Returns: Bmatch, zyx_sorted
            Bmatch: resulting binary image from matching
            zyx_sorted: coordinates sorted by tracing
        """
        # tv
        if sigma_tv > 0:
            Stv, Otv = dtvoting.stick3d(B, O, sigma=sigma_tv)
            mask_tv = Stv > np.exp(-1/2)
            Btv = nonmaxsup.nms3d(Stv*mask_tv, Otv)
            Btv = next(iter(imgutil.connected_components(Btv)))[1]
        else:
            Btv = B
            Otv = O

        # matching
        Bmatch = matching.match_spatial_orient(
            Btv, Otv*Btv, Bsac,
            sigma_hessian=sigma_hessian,
            sigma_tv=sigma_extend
        )

        # bounding
        if mask_bound is not None:
            Bmatch = Bmatch * mask_bound

        # ordering
        zyx_sorted = tracing.Tracing(Bmatch, O*Bmatch).sort_points()
        
        return Bmatch, zyx_sorted

    @staticmethod
    def meshrefine(
            zyx, zyx_ref,
            sigma_normal, sigma_mesh, sigma_hull, mask_bound
        ):
        """ calculate surface normal for one divided part
            zyx: coordinates
            zyx_ref: reference zyx for insideness
            sigma_normal: length scale for normal estimation
            sigma_mesh: spatial resolution for poisson reconstruction
            sigma_hull: length to expand in the normal direction when computing hull
            mask_bound: a mask for boundary
        Returns: zyx, nxyz
            zyx: sorted zyx coordinates
            nxyz: normals in xyz order
        """
        # refine
        zyx_refine = meshrefine.refine_surface(
            zyx,
            sigma_normal=sigma_normal,
            sigma_mesh=sigma_mesh,
            sigma_hull=sigma_hull,
            mask_bound=mask_bound
        )

        # sort
        if mask_bound is not None:
            shape = mask_bound.shape
        else:
            shape = np.ceil(np.max(zyx, axis=0)).astype(np.int_) + 1
        Bsort = utils.points2pixels(zyx_refine, shape)
        _, Osort = features.features3d(Bsort, 1)
        zyx_sort = tracing.Tracing(Bsort, Osort*Bsort).sort_points()

        # calculate normal
        nxyz = meshrefine.normals_points(
            xyz=utils.reverse_coord(zyx_sort),
            sigma=sigma_normal,
            xyz_ref=zyx_ref[::-1]
        )
        return zyx_sort, nxyz
