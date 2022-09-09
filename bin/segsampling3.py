#!/usr/bin/env python
""" Sampling on membranes: uniform sampling.
"""

import sys
import argparse
import textwrap
import pathlib
import logging
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# from sklearn import decomposition

import etsynseg

class SegSampling:
    def __init__(self):
        """ Init.
        """
        # info
        self.prog = "segsampling"
        self.info = """ Info of the attributes.
        args: Arguments received from the terminal. Length unit is nm.
        steps: Intermediate results. Coordinates: ranged in the clipped tomo, in units of pixels, the order is [z,y,x].
        results: Final results. Coordinates: ranged in the input tomo, in units of pixels, the order is [x,y,z].
        """
        # args: length unit is nm
        self.args = dict(
            mode=None, inputs=None, outputs=None,
            seg_file=None, samp_file=None, tomo_file=None,
            labels=None, spacing_xy=None, spacing_z=None
        )

        # intermediate results: length unit is pixel, coordinates are clipped
        # naming: _subi have the same lengths
        self.steps = dict(
            pixel_nm=None, zyx=None, nzyx=None,
            Ic=None, clip_low=None, shape=None,zyxc=None,
            values=None, mask_sub0=None, mask_sub1=None
        )
        
        # results: coordinates are in the original range
        self.results = dict(
            version=etsynseg.__version__,
            tomo_file=None, pixel_nm=None,
            xyz=None, nxyz=None
        )

        # build parser
        self.build_argparser()

        # logging
        self.timer = etsynseg.miscutil.Timer(return_format="string")
        self.logger = logging.getLogger(self.prog)


    #=========================
    # args
    #=========================
    
    def build_argparser(self):
        """ Build argument parser for inputs. Constructs self.argparser (argparse.ArgumentParser).
        """
        # parser
        description = textwrap.dedent("""
        Sampling on membranes: uniform sampling.
        
        Usage:
            segsampling3.py run name-seg.npz -t tomo.mrc -o outputs
            segsampling3.py show name-samp.npz
                tomo: if not provided, then use the tomo file in name-seg.npz
                outputs: if not provided, then set as name-seg-sampling
        """)

        parser = argparse.ArgumentParser(
            prog="segsampling3.py",
            description=description,
            formatter_class=etsynseg.miscutil.HelpFormatterCustom
        )
        # input/output
        parser.add_argument("mode", type=str, choices=["run", "show"])
        parser.add_argument("inputs", type=str, help="Segmentation file for run mode; Sampling file for show mode.")
        parser.add_argument("-t", "--tomo_file", type=str, default=None, help="Tomo file. Defaults to the one in seg_file.")
        parser.add_argument("-o", "--outputs", type=str, default=None, help="Basename for output files. Defaults to 'seg_file-membno'.")

        # options
        parser.add_argument("--label", type=int, default=1, help="The label of segment component to be sampled on.")
        parser.add_argument("--spacing_xy", type=float, default=5, help="Spacing (in nm) between sampling boxes in xy direction.")
        parser.add_argument("--spacing_z", type=float, default=3, help="Spacing (in nm) between sampling boxes in z direction.")
        
        # assign to self
        self.argparser = parser
    
    def load_args(self, args, argv=None):
        """ Load args from seg file.

        Updated attributes:
        args: seg_file, tomo_file, outputs, label, box_localmax
        steps: pixel_nm, zyx, nzyx

        Args:
            args (dict): Args as a dict.
            argv (list): Results of sys.argv, for logging commands used.
        """
        # a helper to set args
        def set_arg(key, default=None):
            if args[key] is not None:
                self.args[key] = args[key]
            else:
                self.args[key] = default

        # set seg filename, load seg
        self.args["seg_file"] = args["inputs"]
        seg = etsynseg.segbase.SegBase().load_state(self.args["seg_file"])
        
        # set tomo filename from arg or seg
        set_arg("tomo_file", seg.args["tomo_file"])
        # set outputs name
        outputs_default = pathlib.Path(
            self.args["seg_file"]).stem.replace("-seg", "-samp")
        set_arg("outputs", outputs_default)
        # set label
        set_arg("label")
        # set spacing
        set_arg("spacing_xy")
        set_arg("spacing_z")

        # get data from seg
        # pixel size
        self.steps["pixel_nm"] = seg.steps["tomod"]["pixel_nm"]
        # load points and normals
        self.steps["zyx"] = etsynseg.pcdutil.reverse_coord(
            seg.results[f"xyz{self.args['label']}"]
        )
        self.steps["nzyx"] = etsynseg.pcdutil.reverse_coord(
            seg.results[f"nxyz{self.args['label']}"]
        )

        # setup logging
        log_handler = logging.FileHandler(self.args["outputs"]+".log")
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel("INFO")
        # initial log
        self.logger.info(f"----{self.prog}----")
        if argv is not None:
            self.logger.info(f"command: {' '.join(argv)}")
        self.logger.info(f"""outputs: {self.args["outputs"]}""")


    #=========================
    # io
    #=========================
    
    def load_state(self, samp_file):
        """ Load info from state file.

        Args:
            samp_file (str): Filename of the state file.
        """
        state = np.load(samp_file, allow_pickle=True)
        self.args = state["args"].item()
        self.steps = state["steps"].item()
        self.results = state["results"].item()
        return self

    def backup_file(self, filename):
        """ If file exists, rename to filename~
        """
        p = pathlib.Path(filename)
        if p.is_file():
            p.rename(filename+"~")

    def save_state(self, samp_file, compress=False, backup=False):
        """ Save data to state file.

        State file keys: args,steps,results

        Args:
            samp_file (str): Filename of the state file.
            compress (bool): Whether to compress the npz. Compression requires more time.
            backup (bool): Whether to backup samp_file if it exists.
        """
        if backup:
            self.backup_file(samp_file)

        if compress:
            func_save = np.savez_compressed
        else:
            func_save = np.savez

        func_save(
            samp_file,
            args=self.args,
            steps=self.steps,
            results=self.results
        )

    #=========================
    # steps
    #=========================

    def load_tomo_clip(self, rescale=False):
        """ Load tomo and clip.

        Updated attributes:
        steps: I, shape, clip_low, zyxc

        Args:
            rescale (bool): Whether to rescale tomo to range 1 to 0.
        """
        # set margin, add some additional space
        margin_nm = 5
        # load tomo, clip
        Ic, clip_low, _ = etsynseg.modutil.read_tomo_clip(
            self.args["tomo_file"], self.steps["zyx"],
            margin_nm=margin_nm, pixel_nm=self.steps["pixel_nm"]
        )

        # rescale to 1-0
        if rescale:
            Ic = etsynseg.imgutil.scale_minmax(
                Ic, qrange=(0.02, 0.98), vrange=(1, 0)
            )

        # set tomo, clip coordinates
        self.steps["Ic"] = Ic
        self.steps["shape"] = Ic.shape
        self.steps["clip_low"] = clip_low
        self.steps["zyxc"] = self.steps["zyx"] - clip_low

    def remove_tomo(self):
        """ Remove tomo (self.steps["Ic"]) to save space.
        """
        self.steps["Ic"] = None

    def reload_tomo(self, tomo_file=None):
        """ Reload tomo and clip. Assign to self.steps["Ic"]. Skip if tomo is already loaded.

        Args:
            tomo_file (str): Filename of tomo. If None, then use self.args["tomo_file"].
        """
        # skip if tomo exists
        if self.steps["Ic"] is not None:
            return
        # setup tomo_file
        if tomo_file is None:
            tomo_file = self.args["tomo_file"]
        # read tomo, clip properly
        clip_low = self.steps["clip_low"]
        clip_high = clip_low + self.steps["shape"]
        Ic, _ = etsynseg.io.read_tomo(
            tomo_file, clip_low=clip_low, clip_high=clip_high
        )
        # assign
        self.steps["Ic"] = Ic

    def sampling_uniform(self):
        """ Averaging for each point. Find per-xy local max.

        Updated attributes:
        steps: values, mask_sub0, mask_sub1
        """
        # load info
        px = self.steps["pixel_nm"]
        # Ic = self.steps["Ic"]
        zyxc = self.steps["zyxc"]
        # nzyx = self.steps["nzyx"]
        
        # spacing in pixel, minimum 1
        spacing_xy = max(1, int(np.round(self.args["spacing_xy"] / px)))
        spacing_z = max(1, int(np.round(self.args["spacing_z"] / px)))
        
        # set values
        # make sampled points a large value + random number to break degeneracy
        values = np.ones(len(zyxc), dtype=int)
        z_arr = np.round(zyxc[:, 0]).astype(int)
        for z in np.unique(z_arr)[np.random.randint(0, spacing_z)::spacing_z]:
            mask_z = z_arr==z
            values_z = np.ones(np.sum(mask_z))
            values_z[np.random.randint(0, spacing_xy)::spacing_xy] = 10
            mask_zxy = values_z==10
            values_z[mask_zxy] += np.random.rand(np.sum(mask_zxy))
            values[mask_z] = values_z
        self.steps["values"] = values

        # per-slice localmax
        # r_thresh: 1 px as the min separation
        mask_sub0 = etsynseg.memsampling.localmax_perslice(
            zyxc, values, r_thresh=1, include_eq=False
        )
        self.steps["mask_sub0"] = mask_sub0

    def final_results(self):
        """ Update attributes in self.results.
        """
        # info
        self.results["pixel_nm"] = self.steps["pixel_nm"]
        self.results["tomo_file"] = self.args["tomo_file"]

        # selected points
        def subset(arr):
            for i in range(1):
                arr = arr[self.steps[f"mask_sub{i}"]]
            return arr
        self.results["xyz"] = etsynseg.pcdutil.reverse_coord(
            subset(self.steps["zyx"])
        )
        self.results["nxyz"] = etsynseg.pcdutil.reverse_coord(
            subset(self.steps["nzyx"])
        )


    #=========================
    # outputs, show
    #=========================    

    def show_steps(self):
        """ Visualize each step as 3d image using etsynseg.plot.imshow3d.
        """
        # setup
        self.reload_tomo()
        steps = self.steps
        # set vector length from box geometry
        vec_length = self.args["spacing_xy"]/steps["pixel_nm"]

        # preprocess tomo
        Ic = steps["Ic"]
        Ic = np.clip(Ic, np.quantile(Ic, 0.02), np.quantile(Ic, 0.98))

        # collect points and vectors
        vecs_zyx = []
        vecs_dir = []
        name_vecs = []
        cmap_vecs = []
        # append data to vecs arrays
        def vecs_append(mask_seq, name, cmap, show_dir=False):
            # mask_seq: e.g. zyxc_sub = zyxc[mask_seq[0]][mask_seq[1]]...
            # add points: subset
            zyxc_sub = steps["zyxc"]
            for mask in mask_seq:
                zyxc_sub = zyxc_sub[mask]
            vecs_zyx.append(zyxc_sub)
            # add vector: subset
            if show_dir:
                nzyx_sub = steps["nzyx"]
                for mask in mask_seq:
                    nzyx_sub = nzyx_sub[mask]
                vecs_dir.append(nzyx_sub)
            else:
                vecs_dir.append(None)
            # add name, cmap
            name_vecs.append(name)
            cmap_vecs.append(cmap)
        # points: segmentation, class1/2, simplified
        vecs_append([], "segmentation", "green")
        vecs_append([steps[f"mask_sub{i}"] for i in range(1)], "sampling boxes", "yellow", show_dir=True)
        
        # show plots
        etsynseg.plot.imshow3d(
            Ic,
            vecs_zyx=vecs_zyx, vecs_dir=vecs_dir,
            vec_size=1, vec_width=2, vec_length=vec_length,
            name_vecs=name_vecs, cmap_vecs=cmap_vecs
        )
        etsynseg.plot.napari.run()


    #=========================
    # workflow
    #=========================

    def workflow(self):
        """ Workflow of sampling.
        """
        # setup
        self.timer.click()

        # sampling
        # load data
        self.load_tomo_clip(rescale=True)
        self.logger.info(f"loaded data: {self.timer.click()}")
        # local max
        self.sampling_uniform()
        self.logger.info(f"finished finding uniform: {self.timer.click()}")
        # results
        self.final_results()

        # outputs
        outputs = self.args["outputs"]
        self.remove_tomo()
        self.save_state(outputs+".npz", backup=True)
        self.logger.info(f"generated outputs: {self.timer.click()}")

        # done
        self.logger.info(f"""segmentation finished: total {self.timer.total()}""")


if __name__ == "__main__":
    # init
    samp = SegSampling()

    # read args
    argv = sys.argv
    args = vars(samp.argparser.parse_args(argv[1:]))
    
    # sampling
    if args["mode"] == "run":
        samp.load_args(args, argv=argv)
        samp.workflow()
    # show
    elif args["mode"] == "show":
        samp.load_state(args["inputs"])
        samp.reload_tomo(tomo_file=args["tomo_file"])
        samp.show_steps()
