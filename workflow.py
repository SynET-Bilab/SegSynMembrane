#!/usr/bin/env python
""" workflow
"""

import numpy as np
from synseg import io, utils, filter
from synseg import hessian, dtvoting, nonmaxsup

__all__ = [
    "MemDetect"
]

class MemDetect:
    def __init__(self):
        """ define attributes
        notations:
            S: float-valued saliency
            O: float-valued orientation, in (-pi/2,pi/2)
            N: int-valued nonmax-suppressed image
        """
        # info of image
        self.I = None
        self.voxel_size = None
        self.clip_range = None
        self.mask = None

        # saliency
        self.S = None
        self.Stv = None
        self.Ntv = None
        self.Nref = None
        self.Nfilt = None
        self.Ssup = None
        self.Nsup = None
        self.Osup = None

    def clean_intermediate(self):
        """ clean intermediate results
        """
        self.S = None
        self.Stv = None
        self.Ntv = None
        self.Nref = None
        self.Nfilt = None
        self.Ssup = None

    def load_tomo(self, tomo_mrc, bound_mod, voxel_size=None, obj=1):
        """ load from raw tomo and model files
        :param tomo_mrc, bound_mod: filename of tomo, boundary
        :param voxel_size: manually set (in nm), or read from mrc
        :param obj: specify object id for boundary in model file
        :return: None
            actions(main): set I, voxel_size, clip_range, mask
        """
        # load tomo and clip
        I, model, voxel_size_A, clip_range = io.read_clip_tomo(
            tomo_mrc, bound_mod
        )
        # mask
        mask = io.model_to_mask(
            model[model["object"] == obj], I[0].shape)
        
        # voxel_size in nm
        if voxel_size is None:
            voxel_size = voxel_size_A.tolist()[0] / 10

        # assign to self
        self.I = I
        self.voxel_size = voxel_size
        self.clip_range = clip_range
        self.mask = mask
    
    def save_result(self, npzname):
        """ save results
        fields: I, voxel_size, clip_range, Nsup, Osup
        """
        results = dict(
            I=self.I.astype(np.int8),
            voxel_size=self.voxel_size,
            clip_range=self.clip_range,
            Nsup=self.Nsup.astype(np.float32),
            Osup=self.Osup.astype(np.float32)
        )
        np.savez(npzname, results)
    
    def load_result(self, npzname):
        """ load results
        fields: I, voxel_size, clip_range, Nsup, Osup
        """
        results = np.load(npzname, allow_pickle=True)
        self.I = results["I"]
        self.voxel_size = results["voxel_size"].item()
        self.clip_range = results["clip_range"].item()
        self.Nsup = results["Nsup"]
        self.Osup = results["Osup"]
    
    def detect(self, sigma_gauss, sigma_tv, sigma_supp,
        qfilter=0.25, min_size=5):
        """ detect membrane features
        :param sigma_gauss, sigma_tv, sigma_supp: sigma's in nm
            sigma_gauss: gaussian smoothing before hessian
            sigma_tv: tensor field for tensor voting
            sigma_supp: tensor field for normal suppression
        :param qfilter: quantile to filter out by Stv and delta_Oref
        :param min_size: min size of segments
        :return: None
            actions(main): set Nsup, Osup
            actions(intermediate): set S,Stv,Ntv,Nref,Nfilt,Ssup
        """
        # negate, hessian
        Ineg = utils.negate_image(self.I)
        sigma_gauss = sigma_gauss / self.voxel_size
        S, O = hessian.features3d(Ineg, sigma_gauss)
        
        # tv, nms
        sigma_tv = sigma_tv / self.voxel_size
        Stv, Otv = dtvoting.stick3d(
            S*self.mask, O*self.mask, sigma_tv
        )
        Ntv = self.mask*nonmaxsup.nms3d_gw(Stv, Otv)
        
        # refine: orientation, nms
        _, Oref = hessian.features3d(Ntv, sigma=1)
        Nref = nonmaxsup.nms3d_gw(Stv*Ntv, Oref)

        # filter: S value, orientation change, len of segment
        # orientation change after refinement
        dOref = np.abs(Oref-Otv)  # in (0, pi)
        mask = dOref > np.pi/2
        dOref[mask] = np.pi - dOref[mask]  # in (0, pi/2)
        # filter out small Stv or large dOref
        Nfilt = filter.filter_connected(
            Nref, [Stv, -dOref], qfilter=qfilter
        )

        # normal suppression
        sigma_supp = sigma_supp / self.voxel_size
        Nsup, Ssup = dtvoting.suppress_by_orient(
            Nfilt, Oref*Nfilt, sigma=sigma_supp, dO_threshold=np.pi/4
        )
        Osup = Oref*Nsup

        # assign to self
        self.S = S
        self.Stv = Stv
        self.Ntv = Ntv
        self.Nref = Nref
        self.Nfilt = Nfilt
        self.Ssup = Ssup
        self.Nsup = Nsup
        self.Osup = Osup
