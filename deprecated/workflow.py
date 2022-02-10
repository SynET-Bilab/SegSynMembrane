# """ workflow
# """

# import numpy as np
# from TomoSynSegAE import io, utils
# from TomoSynSegAE import hessian, dtvoting, nonmaxsup

# __all__ = [
#     "MemDetect"
# ]

# class MemDetect:
#     def __init__(self):
#         """ define attributes
#         notations:
#             S: float-valued saliency
#             O: float-valued orientation, in (-pi/2,pi/2)
#             N: int-valued nonmax-suppressed image
#         """
#         # info of image
#         self.I = None
#         self.voxel_size = None
#         self.voxel_size_nm = None
#         self.clip_range = None
#         self.mask = None

#         # saliency
#         self.S = None
#         self.Stv = None
#         self.Ntv = None
#         self.Nsup = None
#         self.Nfilt = None
#         self.Ofilt = None

#     def clean_intermediate(self):
#         """ clean intermediate results
#         """
#         self.S = None
#         self.Stv = None
#         self.Ntv = None
#         self.Nsup = None

#     def load_tomo(self, tomo_mrc, bound_mod, voxel_size=None, obj=1):
#         """ load from raw tomo and model files
#         :param tomo_mrc, bound_mod: filename of tomo, boundary
#         :param voxel_size: manually set (in A), or read from mrc
#         :param obj: specify object id for boundary in model file
#         :return: None
#             actions(main): set I, voxel_size, clip_range, mask
#         """
#         # load tomo and clip
#         I, model, voxel_size_A, clip_range = io.read_clip_tomo(
#             tomo_mrc, bound_mod
#         )
#         # mask
#         mask = io.model_to_mask(
#             model[model["object"] == obj], I[0].shape)
        
#         # voxel_size in nm
#         if voxel_size is None:
#             voxel_size = voxel_size_A.tolist()[0]

#         # assign to self
#         self.I = I
#         self.voxel_size = voxel_size
#         self.voxel_size_nm = voxel_size / 10
#         self.clip_range = clip_range
#         self.mask = mask
    
#     def save_result(self, npzname):
#         """ save results
#         fields: I, voxel_size, clip_range, Nfilt, Ofilt
#         """
#         results = dict(
#             I=self.I.astype(np.int8),
#             voxel_size=self.voxel_size,
#             clip_range=self.clip_range,
#             mask=self.mask,
#             Nfilt=self.Nfilt.astype(np.float32),
#             Ofilt=self.Ofilt.astype(np.float32)
#         )
#         np.savez_compressed(npzname, **results)
    
#     def load_result(self, npzname):
#         """ load results
#         fields: I, voxel_size, clip_range, Nfilt, Ofilt
#         """
#         results = np.load(npzname, allow_pickle=True)
#         self.I = results["I"]
#         self.voxel_size = results["voxel_size"].item()
#         self.voxel_size_nm = self.voxel_size / 10
#         self.clip_range = results["clip_range"].item()
#         self.mask = results["mask"]
#         self.Nfilt = results["Nfilt"]
#         self.Ofilt = results["Ofilt"]
    
#     def detect(self, sigma_gauss, sigma_tv, sigma_supp,
#         qfilter=0.25, dzfilter=5, min_size=5):
#         """ detect membrane features
#         :param sigma_gauss, sigma_tv, sigma_supp: sigma's in nm
#             sigma_gauss: gaussian smoothing before hessian
#             sigma_tv: tensor field for tensor voting
#             sigma_supp: tensor field for normal suppression
#         :param qfilter: quantile to filter out by Stv and delta_Oref
#         :param min_size: min size of segments
#         :return: None
#             actions(main): set Nfilt,Ofilt
#             actions(intermediate): set S,Stv,Ntv,Nref,Nsup
#         """
#         # negate, hessian
#         Ineg = utils.negate_image(self.I)
#         sigma_gauss = sigma_gauss / self.voxel_size_nm
#         S, O = hessian.features3d(Ineg, sigma_gauss)
        
#         # tv, nms
#         sigma_tv = sigma_tv / self.voxel_size_nm
#         Stv, Otv = dtvoting.stick3d(
#             S*self.mask, O*self.mask, sigma_tv
#         )
#         Ntv = self.mask*nonmaxsup.nms3d(Stv, Otv)
        
#         # refine: orientation, nms, orientation changes
#         _, Oref = hessian.features3d(Ntv, sigma=sigma_gauss)
#         Nref = nonmaxsup.nms3d(Stv*Ntv, Oref)
#         dOref = utils.absdiff_orient(Oref, Otv)

#         # normal suppression
#         sigma_supp = sigma_supp / self.voxel_size_nm
#         Nsup, Ssup = dtvoting.suppress_by_orient(
#             Nref, Oref*Nref, sigma=sigma_supp, dO_threshold=np.pi/4
#         )

#         # filter in xy: small Stv, large dOref, small Ssup
#         Nfilt_xy = utils.filter_connected_xy(
#             Nsup, [Stv, -dOref, Ssup],
#             connectivity=2, stats="median",
#             qfilter=qfilter, min_size=min_size
#         )

#         # filter in 3d: delta z
#         Nfilt_dz = utils.filter_connected_dz(
#             Nfilt_xy, dzfilter=dzfilter, connectivity=2)

#         # assign to self
#         self.S = S
#         self.Stv = Stv
#         self.Ntv = Ntv
#         self.Nsup = Nsup
#         self.Nfilt = Nfilt_dz
#         self.Ofilt = Oref*Nfilt_dz
