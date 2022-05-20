""" 
"""

import time, pathlib
import numpy as np
import napari
import etsynseg

__all__ = [
    "Timer", "SegBase"
]

class Timer:
    """ A timer class.

    Examples:
        timer = Timer(format="number")
        dt_since_last = timer.click()
        dt_since_init = timer.total()
    """
    def __init__(self, format="string"):
        """ Init and record current time.

        Args:
            format (str): Format for returned time difference, "string" or "number".
        """
        self.format = format
        self.t_init = time.perf_counter()
        self.t_last = time.perf_counter()

    def click(self):
        """ Record current time and calc time difference with the last click.

        Returns:
            del_t (str): 
        """
        t_curr = time.perf_counter()
        del_t = t_curr - self.t_last
        self.t_last = t_curr
        if self.format == "string":
            del_t = f"{del_t:.1f}s"
        return del_t
    
    def total(self):
        """ Calc time difference between current and the initial.
        """
        t_curr = time.perf_counter()
        del_t = t_curr - self.t_init
        if self.format == "string":
            del_t = f"{del_t:.1f}s"
        return del_t


class SegBase:
    """ Base class for segmentation.

    Attributes:
        info (str): Info of useful attributes.
        args, steps, results (dict): Useful attributes. See self.info.

    Methods:
        # misc
        register_map
        # io
        load_state, backup_file, save_state
        # outputs
        output_model, output_slices
        # visualization
        show_steps, show_pcds
        # steps
        load_tomod, detect, fit_refine
    """
    def __init__(self):
        """ Init. Example attributes.
        """
        # logging
        self.timer = None
        self.logger = None
        # map
        self.func_map = map
        
        # info
        self.info = """ Info of the attributes.
        args: Arguments received from the terminal. Length unit is nm.
        steps: Intermediate results in each step. Coordinates: ranged in the clipped tomo, in units of pixels, the order is [z,y,x].
        results: Final results. Coordinates: ranged in the input tomo, in units of pixels, the order is [x,y,z].
        """
        # labels
        self.labels = (1,)
        # args: length unit is nm
        self.args = dict(
            mode=None, inputs=None, outputs=None,
            pixel=None, extend=None, neigh_thresh=None,
            detect_gauss=None, detect_tv=None, detect_filt=None, detect_supp=None,
            moosac_lengrids=None, moosac_shrinkside=None, moosac_popsize=None, moosac_tol=None, moosac_maxiter=None
        )
        # intermediate steps: length unit is pixel, coordinates are clipped
        self.steps = dict(
            tomod=dict(
                I=None, shape=None, pixel_nm=None,
                model=None, clip_low=None,
                bound=None, bound_plus=None, bound_minus=None,
                guide=None, normal_ref=None
            ),
            detect=dict(zyx_nofilt=None, zyx=None),
            components=dict(zyx1=None),
            moosac=dict(mpopz1=None, zyx1=None),
            match=dict(zyx1=None),
            meshrefine=dict(zyx1=None)
        )
        # results: coordinates are in the original range
        self.results=dict(
            xyz1=None, nxyz1=None, area1_nm2=None
        )

    def register_map(self, func_map=map):
        """ Register map function for multiprocessing.

        Args:
            func_map (Callable): Map function.
        """
        self.func_map = func_map


    #=========================
    # io
    #=========================
    
    def load_state(self, state_file):
        """ Load info from state file.

        Args:
            state_file (str): Filename of the state file.
        """
        state = np.load(state_file, allow_pickle=True)
        self.args = state["args"].item()
        self.steps = state["steps"].item()
        self.results = state["results"].item()
        return self

    def backup_file(self, filename):
        """ if file exists, rename to filename~
        """
        p = pathlib.Path(filename)
        if p.is_file():
            p.rename(filename+"~")

    def save_state(self, state_file, backup=False):
        """ Save data to state file.

        Args:
            state_file (str): Filename of the state file.
            backup (bool): Whether to backup state_file if it exists.
        """
        if backup:
            self.backup_file(state_file)

        np.savez_compressed(
            state_file,
            info=self.info,
            args=self.args,
            steps=self.steps,
            results=self.results
        )


    #=========================
    # outputs
    #=========================
    
    def output_model(self, model_file, step="meshrefine", labels=None):
        """ Output segmentation to a model file.

        Args:
            step (str): Name of the step, usually "meshrefine".
            model_file (str): Filename for saving the model.
            labels (tuple of int): Choose components to output. E.g. (1,) for zyx1.
        """
        if labels is None:
            labels = self.labels

        # list of points
        # contour of bound
        tomod = self.steps["tomod"]
        B_bound = etsynseg.pcdutil.points2pixels(tomod["bound"], tomod["shape"])
        contour_bound = etsynseg.imgutil.component_contour(B_bound)
        # components
        zyx_segs = [self.steps[step][f"zyx{i}"] for i in labels]

        # shift, combine
        zyx_low = tomod["clip_low"]
        zyx_arr = [
            zyx_i+zyx_low for zyx_i in
            [contour_bound, *zyx_segs]
        ]

        # write model
        etsynseg.io.write_points(
            model_file, zyx_arr,
            break_contour=tomod["neigh_thresh"]*2
        )

    def output_slices(self, fig_file, step="meshrefine", labels=None, nslice=5, dpi=300):
        """ Plot slices of the segmentation, and save.

        Args:
            fig_file (str): Filename for saving the figure.
            step (str): Name of the step.
            labels (tuple of int): Choose components to output. E.g. (1,) for zyx1.
            nslice (int): The number of slices to plot.
            dpi (int): Figure's dpi.
        """
        if labels is None:
            labels = self.labels
        
        # list of points
        # contour of bound
        tomod = self.steps["tomod"]
        B_bound = etsynseg.pcdutil.points2pixels(tomod["bound"], tomod["shape"])
        contour_bound = etsynseg.imgutil.component_contour(B_bound)
        # components
        zyx_segs = [self.steps[step][f"zyx{i}"] for i in labels]
        zyx_segs = [self.steps[step][f"zyx{i}"] for i in (1, 2)]
        # combine
        zyx_arr = [contour_bound, *zyx_segs]

        # generate im_dict for plot.imoverlay
        iz_clip = tomod["clip_low"][0]
        iz_min = np.min(contour_bound[:, 0])
        iz_max = np.max(contour_bound[:, 0])
        iz_arr = np.linspace(iz_min, iz_max, nslice, dtype=int)
        # im dict
        I = self.steps["tomod"]["I"]
        im_dict = {
            # label z in range of original tomo
            f"z={iz+iz_clip}": {
                "I": I[iz],
                "yxs": [zyx_i[zyx_i[:,0]==iz][:, 1:] for zyx_i in zyx_arr]
            }
            for iz in iz_arr
        }

        # plot
        fig, axes = etsynseg.plot.imoverlay(
            im_dict, dpi=dpi, save=fig_file
        )
        return fig, axes

    # def output_membrano(self, fig_file, step="meshrefine", labels=(1,)):
    #     """ avg membranogram
    #     :return: p_mem, p_pick, v_avg
    #     """
    #     # get data
    #     tomo = data["tomo"]
    #     px_nm = data["px_nm"]
    #     pre_zyx = data["pre_zyx"]
    #     pre_nzyx = data["pre_nzyx"]
    #     post_zyx = data["post_zyx"]
    #     post_nzyx = data["post_nzyx"]

    #     # membranogram
    #     _, v_pre = etsynseg.membranogram.interpolate_dist(
    #         zyx_i, 
    #         pre_zyx, pre_nzyx, dist_arr_nm/px_nm, tomo
    #     )
    #     _, v_post = membranogram.interpolate_avg(
    #         post_zyx, post_nzyx, dist_arr_nm/px_nm, tomo
    #     )

    #     # projection
    #     mem_zyx = np.concatenate([pre_zyx, post_zyx], axis=0)
    #     mem_nzyx = np.concatenate([-pre_nzyx, post_nzyx], axis=0)
    #     proj = membranogram.Project().fit(mem_zyx, mem_nzyx)
    #     del mem_zyx, mem_nzyx
    #     p_pre = proj.transform(pre_zyx)
    #     p_post = proj.transform(post_zyx)
    #     e1 = proj.e1
    #     e2 = proj.e2

    #     # calculate angles
    #     e1_orient = np.rad2deg(np.arctan2(e1[1], e1[0]))

    #     # rescale values
    #     v_pre = skimage.exposure.rescale_intensity(
    #         v_pre,
    #         in_range=tuple(np.quantile(v_pre, qrange))
    #     )
    #     v_post = skimage.exposure.rescale_intensity(
    #         v_post,
    #         in_range=tuple(np.quantile(v_post, qrange))
    #     )

    #     # plot
    #     fig, axes, s_pt = plot.scatter(
    #         [np.transpose(p)*px_nm for p in [p_pre, p_post]],
    #         v_arr=[v_pre, v_post],
    #         cmap="gray", shape=(2, 1),
    #     )

    #     axes[0, 0].set(ylabel="pre")
    #     axes[1, 0].set(ylabel="post")
    #     fig.supxlabel(rf"$e_1(\theta_{{xy}}={e1_orient:.1f}^o)/nm$")
    #     fig.supylabel(f"$e_2(z)/nm$")

    #     if save is not None:
    #         fig.savefig(save)
    
    #=========================
    # show
    #=========================

    def show_steps(self, labels=None):
        """ Visualize each step as 3d image using etsynseg.plot.imshow3d

        Args:
            labels (tuple of int): Choose components to output. E.g. (1,) for zyx1.
        """
        if labels is None:
            labels = self.labels
        
        steps = self.steps
        tomod = self.steps["tomod"]
        
        # image
        I = tomod["I"]
        name_I = "clipped tomo"

        # steps setup
        Is_overlay = []
        name_Is = []
        cmap_Is = []
        visible_Is = []

        def im_append(zyx, name, cmap, visible=False):
            if zyx is not None:
                B = etsynseg.pcdutil.points2pixels(zyx, tomod["shape"])
                Is_overlay.append(B)
                name_Is.append(name)
                cmap_Is.append(cmap)
                visible_Is.append(visible)
        
        # add steps
        im_append(tomod["bound"], "bound", "bop blue")
        im_append(steps["detect"]["zyx_nofilt"], "detect(nofilt)", "red", True)
        im_append(steps["detect"]["zyx"], "detect", "bop orange", True)
        for i in labels:
            im_append(steps["components"][f"zyx{i}"], f"components{i}", "magenta")
        for i in labels:
            im_append(steps["moosac"][f"zyx{i}"], f"moosac{i}", "green")
        for i in labels:
            im_append(steps["match"][f"zyx{i}"], f"match{i}", "cyan")
        for i in labels:
            im_append(steps["meshrefine"][f"zyx{i}"], f"meshrefine{i}", "yellow")

        # imshow
        etsynseg.plot.imshow3d(
            I, Is_overlay,
            name_I=name_I, name_Is=name_Is,
            cmap_Is=cmap_Is, visible_Is=visible_Is
        )
        # run napari: otherwise the window will not sustain
        napari.run()

    def show_pcds(self, labels=None):
        """ Draw segmentation as pointclouds.

        Args:
            labels (tuple of int): Choose components to output. E.g. (1,) for zyx1.
        """
        if labels is None:
            labels = self.labels
        
        pts_arr = [self.results[f"xyz{i}"] for i in labels]
        normals_arr = [self.results[f"nxyz{i}"] for i in labels]
        etsynseg.plot.draw_pcds(pts_arr, normals_arr, saturation=1)

    #=========================
    # steps
    #=========================
    
    def load_tomod(self, interp_degree=2):
        """ Load tomo and model.
        
        Prerequisites: args are read.
        Effects: updates self.steps["tomod"].

        Args:
            interp_degree (int): Degree for model interpolation. 2 for quadratic, 1 for linear.
        """
        # log
        self.timer.click()

        # read tomo, model
        args = self.args
        tomod = etsynseg.modutil.read_tomo_model(
            tomo_file=args["tomo_file"],
            model_file=args["model_file"],
            extend_nm=args["extend"],
            pixel_nm=args["pixel"],
            interp_degree=interp_degree
        )

        # update neigh thresh:
        # nm to pixel, constrain to >= 1
        tomod["neigh_thresh"] = max(
            1, args["neigh_thresh"]/tomod["pixel_nm"])

        # save
        self.steps["tomod"].update(tomod)

        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""loaded data: {self.timer.click()}""")

    def detect(self):
        """ Detect membrane-candidates from the image.

        Prerequisites: tomod.
        Effects: updates self.steps["detect"].
        """
        # log
        self.timer.click()

        # setup
        args = self.args
        tomod = self.steps["tomod"]
        pixel_nm = tomod["pixel_nm"]

        # detect mem-like structures
        B, _, B_nofilt = etsynseg.detecting.detect_memlike(
            tomod["I"],
            guide=tomod["guide"],
            bound=tomod["bound"],
            sigma_gauss=tomod["detect_smooth"]/pixel_nm,
            sigma_tv=args["detect_tv"]/pixel_nm,
            factor_filt=args["detect_filt"],
            factor_supp=args["detect_supp"],
            return_nofilt=True
        )

        # save results
        self.steps["detect"]["zyx_nofilt"] = etsynseg.pcdutil.pixels2points(
            B_nofilt)
        self.steps["detect"]["zyx"] = etsynseg.pcdutil.pixels2points(B)

        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""finished detecting: {self.timer.click()}""")

    def fit_refine(self, label):
        """ Fit, match, refine a surface.

        Prerequisites: components are extracted.
        Effects: updates self.steps[field], field="moosac","match","meshrefine"

        Args:
            label (int): Name label for the component. 
        """
        # log
        self.timer.click()

        # setup
        args = self.args
        tomod = self.steps["tomod"]
        pixel_nm = tomod["pixel_nm"]
        guide = tomod["guide"]
        zyx = self.steps["components"][f"zyx{label}"]

        # moosac fitting
        len_grids = tuple(l/pixel_nm for l in args["moosac_lengrids"])
        zyx_fit, mpop_state = etsynseg.moosac.robust_fitting(
            zyx, guide,
            len_grids=len_grids,
            shrink_sidegrid=args["moosac_shrinkside"],
            fitness_rthresh=tomod["neigh_thresh"],
            pop_size=args["moosac_popsize"],
            tol=args["moosac_tol"],
            tol_nback=10,
            max_iter=args["moosac_maxiter"],
            func_map=self.func_map
        )
        # save results
        self.steps["moosac"][f"zyx{label}"] = zyx_fit
        self.steps["moosac"][f"mpopz{label}"] = mpop_state
        # log
        self.logger.info(
            f"""finished moosac ({label}): {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])

        # matching
        zyx_match = etsynseg.matching.match_candidate_to_ref(
            zyx, zyx_fit, guide, r_thresh=tomod["neigh_thresh"]
        )
        # save results
        self.steps["match"][f"zyx{label}"] = zyx_match
        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(
            f"""finished matching ({label}): {self.timer.click()}""")

        # meshrefine
        zyx_refine = etsynseg.meshrefine.refine_surface(
            zyx_match,
            sigma_normal=tomod["neigh_thresh"]*2,
            sigma_mesh=tomod["neigh_thresh"]*2,
            sigma_hull=tomod["neigh_thresh"],
            target_spacing=1,
            bound=tomod["bound"]
        )
        # sort
        zyx_refine = etsynseg.pcdutil.sort_pts_by_guide_3d(zyx_refine, guide)
        # save results
        self.steps["meshrefine"][f"zyx{label}"] = zyx_refine
        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(
            f"""finished meshrefine ({label}): {self.timer.click()}""")
