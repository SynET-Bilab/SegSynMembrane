""" workflow
"""

import numpy as np
from synseg import io, utils
from synseg import hessian, dtvoting, nonmaxsup
from synseg import division, evomsac, matching

# __all__ = [
#     "MemDetect"
# ]

class MemSeg:
    def __init__(self, d_mem_nm=5, d_cleft_nm=20):
        """ define attributes
        notations:
            S: float-valued saliency
            O: float-valued orientation, in (-pi/2,pi/2)
            N: int-valued nonmax-suppressed image
        """
        self.d_mem_nm = d_mem_nm
        self.d_cleft_nm = d_cleft_nm

        # info of tomo and model
        self.I = None
        self.voxel_size_nm = None
        self.clip_range = None
        self.mask_bound = None
        self.zyx_pre = None

        # detection
        self.Bdetc = None
        self.Odetc = None
    
        # division
        self.Bdiv1 = None
        self.Bdiv2 = None

    def read_tomo(self, tomo_file, model_file, obj_bound=1, obj_pre=2, voxel_size_nm=None):
        """ load and clip tomo and model
        :param tomo_file, model_file: filename of tomo, model
        :param obj_bound, obj_pre: obj label for boundary and presynapse, begins with 1
        :param voxel_size_nm: manually set; if None then read from tomo_file
        :return: None
        :action: assign I, voxel_size_nm, mask_bound, zyx_pre
        """
        # read model
        model = io.read_model(model_file)
        
        # set the range of clipping
        # use np.floor/ceil -> int to ensure integers
        model_bound = model[model["object"] == obj_bound]
        self.clip_range = {
            i: (int(np.floor(model_bound[i].min())),
                int(np.ceil(model_bound[i].max())))
            for i in ["x", "y", "z"]
        }

        # read tomo, clip to bound
        I, voxel_size_A = io.read_clip_tomo(
            tomo_file, self.clip_range
        )
        self.I = np.asarray(I)

        # voxel_size in nm
        if voxel_size_nm is None:
            # assumed voxel_size_A is the same for x,y,z
            self.voxel_size_nm = voxel_size_A.tolist()[0] / 10
        else:
            self.voxel_size_nm = voxel_size_nm

        # shift model coordinates
        for i in ["x", "y", "z"]:
            model[i] -= self.clip_range[i][0]

        # generate mask for bound (after clipping)
        self.mask_bound = io.model_to_mask(
            model=model[model["object"] == obj_bound],
            yx_shape=self.I[0].shape
        )

        # get coordinates of presynaptic label
        series_pre = model[model["object"] == obj_pre].iloc[0]
        self.zyx_pre = np.array(
            [series_pre[i] for i in ["z", "y", "x"]]
        )

        # scale lengths
        self.d_mem_px = self.d_mem_nm / self.voxel_size_nm
        self.d_cleft_px = self.d_cleft_nm / self.voxel_size_nm
    
    # def save_result(self, npzname):
    #     """ save results
    #     fields: I, voxel_size, clip_range, Nfilt, Ofilt
    #     """
    #     results = dict(
    #         I=self.I.astype(np.int8),
    #         voxel_size=self.voxel_size,
    #         clip_range=self.clip_range,
    #         mask=self.mask,
    #         Nfilt=self.Nfilt.astype(np.float32),
    #         Ofilt=self.Ofilt.astype(np.float32)
    #     )
    #     np.savez_compressed(npzname, **results)
    
    # def load_result(self, npzname):
    #     """ load results
    #     fields: I, voxel_size, clip_range, Nfilt, Ofilt
    #     """
    #     results = np.load(npzname, allow_pickle=True)
    #     self.I = results["I"]
    #     self.voxel_size = results["voxel_size"].item()
    #     self.voxel_size_nm = self.voxel_size / 10
    #     self.clip_range = results["clip_range"].item()
    #     self.mask = results["mask"]
    #     self.Nfilt = results["Nfilt"]
    #     self.Ofilt = results["Ofilt"]
    
    def set_lengthscale(self, length_nm, length_nm_default):
        if length_nm is None:
            length_vx = length_nm_default / self.voxel_size_nm
        else:
            length_vx = length_nm / self.voxel_size_nm
        return length_vx

    def detect(self):
        """ detect membrane features
        :param sigma_gauss, sigma_tv, sigma_supp: sigma's in nm
            sigma_gauss: gaussian smoothing before hessian
            sigma_tv: tensor field for tensor voting
            sigma_supp: tensor field for normal suppression
        :param qfilter: quantile to filter out by Stv and delta_Oref
        :param min_size: min size of segments
        :return: None
            actions(main): set Nfilt,Ofilt
            actions(intermediate): set S,Stv,Ntv,Nref,Nsup
        """
        # negate, hessian
        Ineg = utils.negate_image(self.I)
        S, O = hessian.features3d(Ineg, sigma=self.d_mem_px)
        
        # tv, nms
        Stv, Otv = dtvoting.stick3d(S*self.mask_bound, O*self.mask_bound, sigma=self.d_cleft_px)
        Btv = self.mask_bound*nonmaxsup.nms3d(Stv, Otv)
        
        # refine: orientation, nms, orientation changes
        _, Oref = hessian.features3d(Btv, sigma=self.d_mem_px)
        Bref = nonmaxsup.nms3d(Stv*Btv, Oref*Btv)
        dOref = utils.absdiff_orient(Oref, Otv)
        
        # normal suppression
        Bsupp, Ssupp = dtvoting.suppress_by_orient(
            Bref, Oref*Bref,
            sigma=self.d_cleft_px*5,
            dO_threshold=np.pi/4
        )

        # filter in xy: small Stv, large dOref, small Ssup
        Bfilt_xy = utils.filter_connected_xy(
            Bsupp, [Stv, -dOref, Ssupp],
            connectivity=2, stats="median",
            qfilter=0.25, min_size=1
        )

        # filter in 3d: z
        # z spans at least nz/2
        dzfilter = int(self.I.shape[0]/2)
        Bfilt_dz = utils.filter_connected_dz(
            Bfilt_xy, dzfilter=dzfilter, connectivity=3)

        # assign results
        Bdetect = Bfilt_dz
        Odetect = Oref*Bdetect
        
        return Bdetect, Odetect
    
    def divide(self, B, O, size_ratio_thresh=0.5):
        """ divide detected image into pre-post candidates
        :param B, O: binary image and orientation
        :param size_ratio_thresh: if size2/size1<size_ratio_thresh, consider that membranes are connected
        :return: Bpre, Bpost
        """
        # extract two largest components
        comp_arr = list(utils.extract_connected(
            B, n_keep=2, connectivity=3
        ))
        size1_raw, comp1_raw = comp_arr[0]
        size2_raw, comp2_raw = comp_arr[1]

        # if two membranes seem connected, divide
        size_ratio = size2_raw / size1_raw
        if size_ratio > size_ratio_thresh:
            comp1 = comp1_raw
            comp2 = comp2_raw
        else:
            comps_div = division.divide_connected(
                comp1_raw, O*comp1_raw,
                seg_max_size=int(self.d_cleft_px),
                seg_neigh_thresh=int(self.d_mem_px),
                n_clusters=2
            )
            comp1, comp2 = comps_div[:2]
        
        # identify pre and post membranes
        iz_pre = int(self.zyx_pre[0])
        yx_pre = self.zyx_pre[1:]
        yx1 = utils.mask_to_coord(comp1[iz_pre])
        yx2 = utils.mask_to_coord(comp2[iz_pre])
        dist1 = np.sum((yx1 - yx_pre)**2, axis=1).min()
        dist2 = np.sum((yx2 - yx_pre)**2, axis=1).min()

        if dist1 < dist2:
            return comp1, comp2
        else:
            return comp2, comp1

    def evomsac_match(self, Bdiv, Odiv, grid_z_nm=50, grid_xy_nm=150):
        # evomsac

        # set grid numbers
        zyx = utils.mask_to_coord(Bdiv)
        l_z = np.ptp(zyx[:, 0]) * self.voxel_size_nm
        n_uz = max(3, int(np.round(l_z/grid_z_nm)+1))
        l_y = np.ptp(zyx[:, 1]) * self.voxel_size_nm
        l_x = np.ptp(zyx[:, 2]) * self.voxel_size_nm
        l_xy = np.linalg.norm([l_x, l_y])
        n_vxy = max(4, int(np.round(l_xy/grid_xy_nm))+1)
        
        # evolve
        mtools = evomsac.MOOTools(
            Bdiv, n_vxy=n_vxy, n_uz=n_uz, nz_eachu=1, r_thresh=1
        )
        mpop = evomsac.MOOPop(mtools, pop_size=40)
        mpop.init_pop()
        mpop.evolve(max_iter=400, tol=(0.01, 10))


        # match
        
        # fit surface
        indiv = mpop.select_by_hypervolume(mpop.log_front[-1])
        pts_net = mpop.mootools.get_coord_net(indiv)
        nu_eval = matching.find_max_wire(pts_net, axis=0)
        nv_eval = matching.find_max_wire(pts_net, axis=1)
        Bfit, _ = mpop.mootools.fit_surface_eval(
            indiv,
            u_eval=np.linspace(0, 1, 2*int(nu_eval)),
            v_eval=np.linspace(0, 1, 2*int(nv_eval))
        )

        # tv
        Stv, Otv = dtvoting.stick3d(Bdiv, Odiv, sigma=self.d_cleft_px)
        Btv = nonmaxsup.nms3d(Stv, Otv)
        Bmatch = matching.match_spatial_orient(
            Btv, Otv*Btv, Bfit,
            sigma_gauss=self.d_mem_px,
            sigma_tv=self.d_mem_px
        )
        Bmatch = next(iter(utils.extract_connected(Bmatch)))[1]

        # smoothing
        Ssmooth, Osmooth = hessian.features3d(Bmatch, self.d_mem_px)
        Bsmooth = nonmaxsup.nms3d(Ssmooth, Osmooth)

        return Bsmooth

    def segmentation(self):
        Bdetect, Odetect = self.detect()
        
        Bdiv_pre, Bdiv_post = self.divide(Bdetect, Odetect)
        
        Bpre = self.evomsac_match(Bdiv_pre, Odetect*Bdiv_pre)
        Bpost = self.evomsac_match(Bdiv_post, Odetect*Bdiv_post)

        return Bpre, Bpost