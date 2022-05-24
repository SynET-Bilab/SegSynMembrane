""" Segmentation base functions.
"""

import pathlib
import argparse
import textwrap
import logging
import numpy as np
import matplotlib.pyplot as plt
import napari
import etsynseg

__all__ = [
    "Timer", "HelpFormatterCustom",
    "SegBase"
]


class HelpFormatterCustom(argparse.ArgumentDefaultsHelpFormatter):
    # RawDescriptionHelpFormatter
    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))
    
    # MetavarTypeHelpFormatter
    def _get_default_metavar_for_optional(self, action):
        return action.type.__name__

class SegBase:
    """ Base class for segmentation.

    Attributes:
        info (str): Info of useful attributes.
        args, steps, results (dict): Useful attributes. See self.info.

    Methods:
        # funcs
        register_map
        # args
        build_argparse, load_args
        # io
        load_state, backup_file, save_state
        # outputs
        output_model, output_slices
        # visualization
        show_steps, show_segpcds
        # steps
        load_tomod, detect, fit_refine
    """
    def __init__(self, prog, labels=(1,)):
        """ Init. Example attributes.

        Args:
            prog (str): Program name.
            labels (tuple of int): Labels for components.
        """
        # program
        self.prog = prog
        self.labels = labels
        
        # logging
        self.timer = etsynseg.miscutil.Timer(return_format="string")
        self.logger = logging.getLogger(self.prog)

        # map
        self.func_map = map
        
        # info
        self.info = """ Info of the attributes.
        args: Arguments received from the terminal. Length unit is nm.
        steps: Intermediate results in each step. Coordinates: ranged in the clipped tomo, in units of pixels, the order is [z,y,x].
        results: Final results. Coordinates: ranged in the input tomo, in units of pixels, the order is [x,y,z].
        """

        # args: length unit is nm
        self.args = dict(
            mode=None, inputs=None, outputs=None,
            tomo_file=None, model_file=None, outputs_state=None,
            pixel=None, extend=None, neigh_thresh=None,
            detect_smooth=None, detect_tv=None, detect_filt=None, detect_supp=None,
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
        )
        # steps where components are treated separately
        # keys: zyxi (additional mpopzi for moosac)
        for step in ["components", "moosac", "match", "meshrefine"]:
            self.steps[step] = {f"zyx{i}": None for i in self.labels}
        self.steps["moosac"].update(
            {f"mpopz{i}": None for i in self.labels}
        )

        # results: coordinates are in the original range
        # keys: zyxi, nzyxi, areai_nm2
        self.results = {}
        for i in self.labels:
            self.results[f"zyx{i}"] = None
            self.results[f"nzyx{i}"] = None
            self.results[f"area{i}_nm2"] = None

    def register_map(self, func_map=map):
        """ Register map function for multiprocessing.

        Args:
            func_map (Callable): Map function.
        """
        self.func_map = func_map

    #=========================
    # parser
    #=========================
    
    def build_argparser(self):
        """ Build argument parser for inputs. Constructs self.argparser (argparse.ArgumentParser).
        """
        # parser
        description = textwrap.dedent(f"""
        Semi-automatic membrane segmentation.
        
        Usage:
            (u1) {self.prog}.py mode tomo.mrc model.mod -o outputs [options]
                model: if not provided, then set as tomo.mod
                outputs: if not provided, then set as model-seg
            (u2) {self.prog}.py mode state.npz [options]
        
        Modes:
            run (u1): normal segmentation
            runfine (u1): run with finely-drawn model which separates pre and post
            contresults (u2): continue calculating results.
        """)

        parser = argparse.ArgumentParser(
            prog=f"{self.prog}.py",
            description=description,
            formatter_class=etsynseg.segbase.HelpFormatterCustom
        )
        # mode
        parser.add_argument("mode", type=str, choices=[
            "run", "runfine", "contresults"
        ])
        
        # input/output
        parser.add_argument("inputs", type=str, nargs='+',help="Input files. Tomo and model files for modes in (run, runfine). State file for other modes.")
        parser.add_argument("-o", "--outputs", type=str, default=None, help="Basename for output files. Defaults to the basename of model file.")
        
        # basics
        parser.add_argument("-px", "--pixel", type=float, default=None, help="Pixel size in nm. If not set, then read from the header of tomo.")
        parser.add_argument("--extend", type=float, help="The distance (in nm) that the bounding region extends from guiding lines.")
        parser.add_argument("--neigh_thresh", type=float, help="Distance threshold (in nm) for neighboring points in graph construction.")
        
        # detect
        parser.add_argument("--detect_smooth", type=float, help="Step 'detect': sigma for gaussian smoothin in nm. Can be set to membrane thickness.")
        parser.add_argument("--detect_tv", type=float, help="Step 'detect': sigma for tensor voting in nm. Should be smaller than membrane spacings.")
        parser.add_argument("--detect_filt", type=float, help="Step 'detect': keep the strongest (detect_filt * size of guiding surface) pixels during filtering. A larger value keeps more candidates for the next step.")
        parser.add_argument("--detect_supp", type=float, help="Step 'detect': sigma for normal suppression = (detect_supp * length of guiding line).")
        
        # components
        parser.add_argument("--components_min", type=float, help="Step 'components': min size of component = (components_min * size of guiding surface). Raise error if only smaller ones are obtained.")
        
        # moosac
        parser.add_argument("--moosac_lengrids", type=float, nargs=2, help="Step 'moosac': length of sampling grids in z- and xy-axes. More complex surface requires smaller grids.")
        parser.add_argument("--moosac_shrinkside", type=float, help="Step 'moosac': grids on the side in xy are shrinked to this value. A smaller facilicates segmentation towards the bounding region.")
        parser.add_argument("--moosac_resize", type=float, help="Step 'moosac': temporarily increase pixel size (nm) to this value to simplify computation. Can be set to membrane thickness.")
        parser.add_argument("--moosac_popsize", type=int, help="Step 'moosac': population size for evolution.")
        parser.add_argument("--moosac_tol", type=float, help="Step 'moosac': terminate if fitness change < tol in all last 10 steps.")
        parser.add_argument("--moosac_maxiter", type=int, help="Step 'moosac': max number of iterations.")

        # assign to self
        self.argparser = parser

    def load_args(self, args):
        """ Load args into self.args.

        Args:
            args (dict or argparse.Namespace): Args as a dict, or from parser.parse_args.
        """
        # conversion
        if type(args) is not dict:
            args = vars(args)

        # processing
        mode = args["mode"]
        # modes reading tomo and model files
        if mode in ["run", "runfine"]:
            # amend tomo, model
            args["tomo_file"] = args["inputs"][0]
            if len(args["inputs"]) == 2:
                args["model_file"] = args["inputs"][1]
            else:
                args["model_file"] = str(pathlib.Path(
                    args["tomo_file"]).with_suffix(".mod"))
            # amend outputs
            if args["outputs"] is None:
                args["outputs"] = pathlib.Path(
                    args["model_file"]).stem + "-seg"
            # save
            self.args.update(args)
        # modes reading state file
        elif mode in ["contresults"]:
            state_file = args["inputs"][0]
            self.load_state(state_file)
        else:
            raise ValueError(f"""Unrecognized mode: {mode}""")

        # state filename
        self.args["outputs_state"] = self.args["outputs"]+".npz"

        # setup logging
        log_handler = logging.FileHandler(self.args["outputs"]+".log")
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel("INFO")

        # log for modes that run segmentation
        if args["mode"] in ["run", "runfine"]:
            self.logger.info(f"----{self.prog}----")
            self.logger.info(f"""read args: {args["mode"]} {args["inputs"]}""")
            # save state, backup for the first time
            self.save_state(self.args["outputs_state"], backup=True)
            self.logger.info("saved state (with backups)")
        # print for modes that just show
        else:
            print(f"----{self.prog}----")
            print(f"""mode: {args["mode"]}""")


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

    def save_state(self, state_file, compress=False, backup=False):
        """ Save data to state file.

        Args:
            state_file (str): Filename of the state file.
            compress (bool): Whether to compress the npz. No compression saves time
            backup (bool): Whether to backup state_file if it exists.
        """
        if backup:
            self.backup_file(state_file)

        if compress:
            func_save = np.savez_compressed
        else:
            func_save = np.savez

        func_save(
            state_file,
            prog=self.prog,
            info=self.info,
            args=self.args,
            steps=self.steps,
            results=self.results
        )

    #=========================
    # steps
    #=========================
    
    def load_tomod(self, interp_degree=2, raise_noref=False):
        """ Load tomo and model.
        
        Prerequisites: args are read.
        Effects: updates self.steps["tomod"].

        Args:
            interp_degree (int): Degree for model interpolation. 2 for quadratic, 1 for linear.
            raise_noref (bool): Whether to raise error if the reference point is not found in model object 2.
        """
        # log
        self.timer.click()

        # read tomo, model
        args = self.args
        try:
            tomod = etsynseg.modutil.read_tomo_model(
                tomo_file=args["tomo_file"],
                model_file=args["model_file"],
                extend_nm=args["extend"],
                pixel_nm=args["pixel"],
                interp_degree=interp_degree,
                raise_noref=raise_noref
            )
        # FileNotFoundError: no tomo or no model file
        # ValueError: no ref point if raise_noref=True
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(e)
            raise

        # update neigh thresh:
        # nm to pixel, constrain to >= 1
        tomod["neigh_thresh"] = max(args["neigh_thresh"]/tomod["pixel_nm"], 1)

        # save
        self.steps["tomod"].update(tomod)

        # log
        self.logger.info(f"""loaded data: {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])
        self.logger.info("saved state")

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
            zyx_guide=tomod["guide"], B_bound=tomod["bound"],
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
        self.logger.info(f"""finished detecting: {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])
        self.logger.info("saved state")

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
            downscale=args["moosac_resize"]/pixel_nm,
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
        # self.save_state(self.args["outputs_state"])

        # matching
        # r_thresh: at least 2*moosac_resize
        zyx_match = etsynseg.matching.match_candidate_to_ref(
            zyx, zyx_fit, guide,
            r_thresh=max(args["moosac_resize"]/pixel_nm*2, 2)
        )
        # save results
        self.steps["match"][f"zyx{label}"] = zyx_match
        # log
        # self.save_state(self.args["outputs_state"])
        self.logger.info(
            f"""finished matching ({label}): {self.timer.click()}""")

        # meshrefine
        zyx_refine = etsynseg.meshrefine.refine_surface(
            zyx_match,
            sigma_normal=tomod["neigh_thresh"]*2,
            sigma_mesh=tomod["neigh_thresh"]*2,
            sigma_hull=tomod["neigh_thresh"],
            target_spacing=1,
            B_bound=tomod["bound"]
        )
        # sort
        zyx_refine = etsynseg.pcdutil.sort_pts_by_guide_3d(zyx_refine, guide)
        # save results
        self.steps["meshrefine"][f"zyx{label}"] = zyx_refine
        # log
        self.logger.info(
            f"""finished meshrefine ({label}): {self.timer.click()}""")
        self.save_state(self.args["outputs_state"])
        self.logger.info("saved state")

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
        print(f"----{self.prog}----")
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
        im_append(etsynseg.pcdutil.pixels2points(tomod["bound"]), "bound", "bop blue")
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

    def show_segpcds(self, labels=None):
        """ Draw segmentation as pointclouds.

        Args:
            labels (tuple of int): Choose components to output. E.g. (1,) for zyx1.
        """
        if labels is None:
            labels = self.labels

        results = self.results
        pts_arr = []
        normals_arr = []
        for i in labels:
            if (f"xyz{i}" in results) and ((f"nxyz{i}" in results)):
                pts_arr.append(results[f"xyz{i}"])
                normals_arr.append(results[f"nxyz{i}"])

        etsynseg.plot.draw_pcds(pts_arr, normals_arr, saturation=1)
    
    def show_moosac(self, labels=None):
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
            mpop_i.plot_logs(title=f"MOOSAC trajectory for component {i}")
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
        B_bound = tomod["bound"]
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
        B_bound = tomod["bound"]
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
