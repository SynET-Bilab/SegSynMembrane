""" 
"""

import time, pathlib
import numpy as np
import matplotlib.pyplot as plt
import napari
import etsynseg

__all__ = [
    "Timer", "SegBase"
]

class Timer:
    """ A timer class.

    Examples:
        timer = Timer(return_format="number")
        dt_since_last = timer.click()
        dt_since_init = timer.total()
    """
    def __init__(self, return_format="string"):
        """ Init and record current time.

        Args:
            return_format (str): Format for returned time difference, "string" or "number".
        """
        self.return_format = return_format
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
        if self.return_format == "string":
            del_t = f"{del_t:.1f}s"
        return del_t
    
    def total(self):
        """ Calc time difference between current and the initial.
        """
        t_curr = time.perf_counter()
        del_t = t_curr - self.t_init
        if self.return_format == "string":
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
            tomo_file=None, model_file=None, outputs_state=None,
            pixel=None, extend=None, neigh_thresh=None,
            detect_gauss=None, detect_tv=None, detect_filt=None, detect_supp=None,
            components_min=None,
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
            sigma_gauss=args["detect_smooth"]/pixel_nm,
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

    #=========================
    # show
    #=========================

    def show_args(self):
        """ Print self.args.
        """
        args = self.args
        k_shown = []

        # functions for printing
        def print_keys(k_arr, delim=", "):
            doc = []
            for k in k_arr:
                doc.append(f"{k}={args[k]}")
                k_shown.append(k)
            doc = delim.join(doc)
            print(doc)

        def print_prefix(prefix, delim=", "):
            print(f"--{prefix}--")
            doc = []
            for k, v in args.items():
                if k.startswith(prefix):
                    doc.append(f"{k.split(prefix)[-1]}={v}")
                    k_shown.append(k)
            doc = delim.join(doc)
            print(doc)

        def print_misc(title="--misc--", delim=", "):
            doc = []
            for k, v in args.items():
                if k not in k_shown:
                    doc.append(f"{k}={v}")
            if len(doc) > 0:
                doc = delim.join(doc)
                print(title)
                print(doc)

        # print
        print("----arguments----")
        # basics
        print("--input/output--")
        print_keys(["tomo_file", "model_file", "outputs"], delim="\n")
        k_shown.extend(["inputs", "outputs_state"])
        print("--basics--")
        print_keys(["mode", "pixel", "extend", "neigh_thresh"])
        # steps
        print_prefix("detect")
        print_prefix("components")
        print_prefix("moosac")
        # misc
        print_misc("--misc--")

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
            im_append(steps["components"]
                      [f"zyx{i}"], f"components{i}", "magenta")
        for i in labels:
            im_append(steps["moosac"][f"zyx{i}"], f"moosac{i}", "green")
        for i in labels:
            im_append(steps["match"][f"zyx{i}"], f"match{i}", "cyan")
        for i in labels:
            im_append(steps["meshrefine"]
                      [f"zyx{i}"], f"meshrefine{i}", "yellow")

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
    
    def show_moo(self, labels=None):
        """ Plot MOOSAC evolution trajectory.

        Args:
            labels (tuple of int): Choose components to output.
        """
        if labels is None:
            labels = self.labels

        for i in labels:
            mpop_i = etsynseg.moosac.MOOPop().init_from_state(
                self.steps["moosac"][f"mpopz{i}"]
            )
            mpop_i.plot_logs()
            plt.show()

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
        B_bound = etsynseg.pcdutil.points2pixels(
            tomod["bound"], tomod["shape"])
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
        B_bound = etsynseg.pcdutil.points2pixels(
            tomod["bound"], tomod["shape"])
        contour_bound = etsynseg.imgutil.component_contour(B_bound)
        # components
        zyx_segs = [self.steps[step][f"zyx{i}"] for i in labels]
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
                "yxs": [zyx_i[zyx_i[:, 0] == iz][:, 1:] for zyx_i in zyx_arr]
            }
            for iz in iz_arr
        }

        # plot
        fig, axes = etsynseg.plot.imoverlay(
            im_dict, dpi=dpi, save=fig_file
        )
        return fig, axes
