#!/usr/bin/env python
""" Sampling on membranes.
"""

import argparse
import textwrap
import pathlib
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
        self.steps = dict(
            pixel_nm=None, zyx=None, nzyx=None,
            Ic=None, clip_low=None, shape=None,zyxc=None,
            values=None, mask_sub0=None,
            boxes_sub1=None, member_sub2=None,
            mask_sub1=None, mask_sub2=None,
        )
        
        # results: coordinates are in the original range
        self.results = dict(
            version=etsynseg.__version__,
            tomo_file=None, pixel_nm=None,
            xyz=None, nxyz=None
        )

        # build parser
        self.build_argparser()


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
    
    def load_args(self, args):
        """ Load args from seg file.

        Setting up attributes:
        args: seg_file, tomo_file, outputs, label, box_localmax, box_classify
        steps: pixel_nm, zyx, nzyx

        Args:
            args (dict): Args as a dict.
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
        """ if file exists, rename to filename~
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

    def load_tomo_clip(self):
        """ Load tomo and clip.

        Setting up attributes:
        steps: I, shape, clip_low, zyxc
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
        # set tomo, clip coordinates
        self.steps["Ic"] = Ic
        self.steps["shape"] = Ic.shape
        self.steps["clip_low"] = clip_low
        self.steps["zyxc"] = self.steps["zyx"] - clip_low
    
    def remove_tomo(self):
        """ Remove tomo (self.steps["Ic"]) to save space.

        Tomo can be removed after self.detect.
        Reload before self.show_steps or self.output_slices.
        """
        self.steps["Ic"] = None

    def reload_tomo(self, tomo_file=None):
        """ Reload tomo and clip. Skip if tomo is already loaded.

        Setting up self.steps["Ic"]

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

    def sampling(self):
        """ Workflow of membranogram calculation and plotting.
        """
        # setup
        px = self.steps["pixel_nm"]
        zyxc = self.steps["zyxc"]
        nzyx = self.steps["nzyx"]
        # preprocess tomo
        Ic = etsynseg.imgutil.scale_minmax(
            self.steps["Ic"], 
            qrange=(0.02, 0.98), vrange=(1, 0)
        )

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

        # extract boxes
        box_classify = np.asarray(self.args["box_classify"])/px
        boxes_sub1 = etsynseg.memsampling.extract_box_2drot(
            Ic, zyxc[mask_sub0], nzyx[mask_sub0],
            box_rn=box_classify[:2],
            box_rt=box_classify[2],
            normalize=True
        )
        self.steps["boxes_sub1"] = boxes_sub1

        # classification
        member_sub1 = etsynseg.memsampling.classify_2drot_boxes(
            boxes_sub1,
            box_rn=box_classify[:2],
            box_rt=box_classify[2]
        )
        self.steps["member_sub1"] = member_sub1

        # simplify by exclusion
        mask_sub1 = member_sub1==0
        r_thresh = self.args["r_exclude"]/px
        mask_sub2 = etsynseg.memsampling.localmax_exclusion(
            zyxc[mask_sub0][mask_sub1],
            values[mask_sub0][mask_sub1],
            r_thresh=r_thresh
        )
        self.steps["mask_sub1"] = mask_sub1
        self.steps["mask_sub2"] = mask_sub2

    def final_results(self):
        self.results["pixel_nm"] = self.steps["pixel_nm"]
        self.results["tomo_file"] = self.args["tomo_file"]

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

    def workflow(self):
        
        self.load_tomo_clip()
        self.sampling()

        # outputs
        outputs = self.args["outputs"]
        self.remove_tomo()
        self.save_state(outputs+".npz")
        self.output_class(fig_file=outputs+".png")
        self.output_boxes(tomo_file=outputs+".mrc")

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
        boxes = self.steps["boxes_sub1"]
        emb = decomposition.PCA(n_components=2).fit_transform(
            boxes.reshape((len(boxes), -1))
        )
        avg = np.mean(boxes, axis=0)
        # memberships
        member = self.steps["member_sub1"]
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
            label =f"class {i+1}"
            # plot avg
            ax_avg[i].imshow(avg[i], origin="lower")
            ax_avg[i].set(title=label)
            ax_avg[i].axis("off")
            # plot embedding
            ax_emb.scatter(*emb[sub].T, label=f"{label},n={np.sum(sub)}")
        # set ax_emb
        ax_emb.axis("off")
        ax_emb.legend(loc=1, title="PC1,PC2")
        
        # save
        fig.savefig(fig_file)

    def output_boxes(self, tomo_file):
        steps = self.steps
        boxes_sub3 = steps["boxes_sub1"][steps["mask_sub1"]][steps["mask_sub2"]]
        etsynseg.io.write_tomo(
            tomo_file, boxes_sub3, pixel_A=steps["pixel_nm"]*10
        )

    def show_steps(self):
        """ Visualize each step as 3d image using etsynseg.plot.imshow3d

        Args:
            labels (tuple of int): Choose components to output. E.g. (1,) for zyx1.
        """
        # image
        self.reload_tomo()
        
        steps = self.steps
        norm_scale = self.args["box_classify"][1]/steps["pixel_nm"]

        Ic = np.clip(
            steps["Ic"],
            np.quantile(steps["Ic"], 0.02),
            np.quantile(steps["Ic"], 0.98)
        )

        # show pts
        vecs_zyx = []
        vecs_dir = []
        name_vecs = []
        cmap_vecs = []

        def vecs_append(mask_seq, name, cmap, show_dir=False):
            zyxc_sub = steps["zyxc"]
            for mask in mask_seq:
                zyxc_sub = zyxc_sub[mask]
            vecs_zyx.append(zyxc_sub)
            
            if show_dir:
                nzyx_sub = steps["nzyx"]
                for mask in mask_seq:
                    nzyx_sub = nzyx_sub[mask]
                vecs_dir.append(nzyx_sub)
            else:
                vecs_dir.append(None)

            name_vecs.append(name)
            cmap_vecs.append(cmap)

        vecs_append([], "segmentation", "green")
        vecs_append([steps[f"mask_sub{i}"] for i in range(2)], "localmax, particle-like", "bop orange")
        vecs_append([steps["mask_sub0"], steps["member_sub1"]==1], "localmax, membrane-like", "bop blue")
        vecs_append([steps[f"mask_sub{i}"] for i in range(3)], "localmax, excluded", "yellow")

        etsynseg.plot.imshow3d(
            Ic,
            vecs_zyx=vecs_zyx,
            vecs_dir=vecs_dir,
            vec_size=1, vec_width=2,
            vec_length=norm_scale
        )
        etsynseg.plot.napari.run()




if __name__ == "__main__":
    # init
    samp = SegSampling()

    # read args
    args = vars(samp.argparser.parse_args())
    
    if args["mode"] == "run":
        samp.load_args(args)
        samp.workflow()
    elif args["mode"] == "show":
        samp.load_state(args["inputs"])
        samp.reload_tomo(tomo_file=args["tomo_file"])
        samp.show_steps()
