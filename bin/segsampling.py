#!/usr/bin/env python
""" Sampling on membranes.
"""

import sys
import argparse
import textwrap
import pathlib
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import decomposition

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
            labels=None, box_localmax=None, box_classify=None, r_exclude=None
        )

        # intermediate results: length unit is pixel, coordinates are clipped
        # naming: _subi have the same lengths
        self.steps = dict(
            pixel_nm=None, zyx=None, nzyx=None,
            Ic=None, clip_low=None, shape=None,zyxc=None,
            values=None, mask_sub0=None,
            boxes_sub2=None, mask_sub1=None,
            member_sub2=None, mask_sub2=None
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
        Sampling on membranes.
        
        Usage:
            segsampling.py run name-seg.npz -t tomo.mrc -o outputs
            segsampling.py show name-samp.npz
                tomo: if not provided, then use the tomo file in name-seg.npz
                outputs: if not provided, then set as name-seg-sampling
        """)

        parser = argparse.ArgumentParser(
            prog="segsampling.py",
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
        parser.add_argument("--box_localmax", type=float, nargs=3, default=(3, 10, 4), help="Geometry of cylindrical box for local max calculations. 3-tuple for lengths (in nm) in normal/tangent directions starting from the segmentation, (lower end in normal, higher end in normal, radius in tangent). Can be set to slightly larger than the target particle. Can exclude the membrane by raising the lower end in normal.")
        parser.add_argument("--box_classify", type=float, nargs=3, default=None, help="Box geometry for classification. 3-tuple similar to box_localmax. If None, then set to the same as box_localmax.")
        parser.add_argument("--r_exclude", type=float, default=2, help="Radius of exclusion (in nm). The final sampling points are no closer than this value.")
        
        # assign to self
        self.argparser = parser
    
    def load_args(self, args, argv=None):
        """ Load args from seg file.

        Updated attributes:
        args: seg_file, tomo_file, outputs, label, box_localmax, box_classify
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
        # set boxes
        set_arg("box_localmax")
        set_arg("box_classify", args["box_localmax"])
        # set exclusion
        set_arg("r_exclude")

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
        margin_nm = 2 + np.max(
            [*self.args["box_localmax"], *self.args["box_classify"]]
        )
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

    def sampling_localmax(self):
        """ Averaging for each point. Find per-xy local max.

        Updated attributes:
        steps: values, mask_sub0, mask_sub1
        """
        # load info
        px = self.steps["pixel_nm"]
        Ic = self.steps["Ic"]
        zyxc = self.steps["zyxc"]
        nzyx = self.steps["nzyx"]
        
        # values
        box_localmax = np.asarray(self.args["box_localmax"])/px
        values = etsynseg.memsampling.extract_box_avg(
            Ic, zyxc, nzyx,
            box_rn=box_localmax[:2],
            box_rt=box_localmax[2]
        )
        self.steps["values"] = values

        # per-slice localmax
        # r_thresh: 1.5 px as the min separation
        mask_sub0 = etsynseg.memsampling.localmax_perslice(
            zyxc, values, r_thresh=1.5
        )
        self.steps["mask_sub0"] = mask_sub0

        # simplify by exclusion
        r_thresh = self.args["r_exclude"]/px
        mask_sub1 = etsynseg.memsampling.localmax_exclusion(
            zyxc[mask_sub0], values[mask_sub0],
            r_thresh=r_thresh
        )
        self.steps["mask_sub1"] = mask_sub1

    def sampling_classify(self):
        """ Classify local maxes. Simplify by exclusion.

        Updated attributes:
        steps: boxes_sub2, member_sub2, mask_sub2
        """
        # load info
        px = self.steps["pixel_nm"]
        Ic = self.steps["Ic"]
        zyxc = self.steps["zyxc"]
        nzyx = self.steps["nzyx"]
        mask_sub0 = self.steps["mask_sub0"]
        mask_sub1 = self.steps["mask_sub1"]

        # extract boxes
        box_classify = np.asarray(self.args["box_classify"])/px
        boxes_sub2 = etsynseg.memsampling.extract_box_2drot(
            Ic,
            zyxc[mask_sub0][mask_sub1],
            nzyx[mask_sub0][mask_sub1],
            box_rn=box_classify[:2],
            box_rt=box_classify[2],
            normalize=True
        )
        self.steps["boxes_sub2"] = boxes_sub2

        # classification
        member_sub2 = etsynseg.memsampling.classify_2drot_boxes(
            boxes_sub2,
            box_rn=box_classify[:2],
            box_rt=box_classify[2]
        )
        self.steps["member_sub2"] = member_sub2
        self.steps["mask_sub2"] = member_sub2==0


    def final_results(self):
        """ Update attributes in self.results.
        """
        # info
        self.results["pixel_nm"] = self.steps["pixel_nm"]
        self.results["tomo_file"] = self.args["tomo_file"]

        # selected points
        def subset(arr):
            for i in range(3):
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
    
    def output_class(self, fig_file):
        """ Plot classes (average, embedding). Save figure.

        Args:
            fig_file (str): Filename for saving the figure.
        """
        # setup, preprocess
        # boxes
        boxes = self.steps["boxes_sub2"]
        emb = decomposition.PCA(n_components=2).fit_transform(
            boxes.reshape((len(boxes), -1))
        )
        # memberships
        member = self.steps["member_sub2"]
        nclass = len(np.unique(member))

        # plotting
        # set grids, axes
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(nclass, nclass+1)
        ax_avg = [fig.add_subplot(gs[i, 0]) for i in range(nclass)]
        ax_emb = fig.add_subplot(gs[:, 1:])
        # plot each class
        for i in range(nclass):
            # subset class i
            sub = member==i
            label = f"class {i+1}"
            # plot avg
            avg = np.mean(boxes[sub], axis=0)
            ax_avg[i].imshow(avg, origin="lower")
            ax_avg[i].set(title=label)
            ax_avg[i].axis("off")
            # plot embedding
            ax_emb.scatter(
                *emb[sub].T,
                label=f"{label},n={np.sum(sub)}",
                s=5, alpha=0.5
            )
        # set ax_emb
        ax_emb.axis("off")
        ax_emb.legend(loc=1, title="PC1,PC2")
        
        # save
        fig.savefig(fig_file)

    def output_boxes(self, tomo_file):
        """ Save sampling boxes as mrcfile.

        Args:
            tomo_file (str): Filename for saving the boxes.
        """
        steps = self.steps
        boxes_sub2 = steps["boxes_sub2"]
        etsynseg.io.write_tomo(
            tomo_file, boxes_sub2, pixel_A=steps["pixel_nm"]*10
        )

    def show_steps(self):
        """ Visualize each step as 3d image using etsynseg.plot.imshow3d.
        """
        # setup
        self.reload_tomo()
        steps = self.steps
        # set vector length from box geometry
        vec_length = self.args["box_classify"][1]/steps["pixel_nm"]

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
        vecs_append([steps[f"mask_sub{i}"] for i in range(2)], "local max", "blue")
        vecs_append([steps[f"mask_sub{i}"] for i in range(3)], "class 1", "yellow", show_dir=True)
        vecs_append([steps["mask_sub0"], steps["mask_sub1"], steps["member_sub2"]!=0], "other classes", "red")

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
        self.sampling_localmax()
        self.logger.info(f"finished finding localmax: {self.timer.click()}")
        # classify
        self.sampling_classify()
        self.logger.info(f"finished classification: {self.timer.click()}")
        # results
        self.final_results()

        # outputs
        outputs = self.args["outputs"]
        self.remove_tomo()
        self.save_state(outputs+".npz", backup=True)
        self.output_class(fig_file=outputs+".png")
        self.output_boxes(tomo_file=outputs+".mrc")
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
