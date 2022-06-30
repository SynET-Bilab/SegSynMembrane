#!/usr/bin/env python
""" Generate membranograms
"""

import argparse
import textwrap
import pathlib
import numpy as np
import skimage
import etsynseg

class SegMembrano:
    def __init__(self):
        """ Init.
        """
        # build parser
        self.build_argparser()

    def build_argparser(self):
        """ Build argument parser for inputs. Constructs self.argparser (argparse.ArgumentParser).
        """
        # parser
        description = textwrap.dedent(f"""
        Generate membranograms.
        
        Usage:
            segmembrano.py name-seg.npz -t tomo.mrc -o outputs --dists <start> <stop> <step>
                tomo: if not provided, then use the tomo file in name-seg.npz
                outputs: if not provided, then set as name-seg-membno
                dists: average membranograms over a range of distances, range(start,stop,step), from the membrane
        """)

        parser = argparse.ArgumentParser(
            prog="segmembrano.py",
            description=description,
            formatter_class=etsynseg.miscutil.HelpFormatterCustom
        )
        # input/output
        parser.add_argument("seg_file", type=str, help="Segmentation state file, generated from segprepost/segonemem.py.")
        parser.add_argument("-t", "--tomo_file", type=str, default=None, help="Tomo file. Defaults to the one in seg_file.")
        parser.add_argument("-o", "--outputs", type=str, default=None, help="Basename for output files. Defaults to 'seg_file-membno'.")

        # options
        parser.add_argument("--dists", type=float, nargs=3, default=[0,1,1], help="Distances to be evaluated at, formatted as (start,stop,step) in nm, where the stopping point is not included.")
        parser.add_argument("--labels", type=int, nargs='+', default=None, help="Labels of components to be plotted. Defaults to all components.")
        
        # assign to self
        self.argparser = parser
    
    def load_args(self, args):
        """ Load args into self.args.

        Setting up attributes from args:
            pixel_nm,tomo_file,outputs,labels
            dists,zyxs,nzyxs

        Args:
            args (dict): Args as a dict.
        """
        # load seg_file
        seg_file = args["seg_file"]
        seg = etsynseg.segbase.SegBase().load_state(seg_file)
        self.pixel_nm = seg.steps["tomod"]["pixel_nm"]

        # read tomo filename from arg or seg
        if args["tomo_file"] is None:
            self.tomo_file = seg.args["tomo_file"]
        else:
            self.tomo_file = args["tomo_file"]

        # get outputs name
        if args["outputs"] is None:
            self.outputs = pathlib.Path(seg_file).stem + "-membno"
        else:
            self.outputs = args["outputs"]
        
        # labels
        if args["labels"] is None:
            self.labels = seg.labels
        else:
            self.labels = args["labels"]
        
        # distance in pixel
        self.dists = np.arange(*args["dists"])/self.pixel_nm

        # points and normals
        self.zyxs = {
            i: etsynseg.pcdutil.reverse_coord(seg.results[f"xyz{i}"])
            for i in self.labels
        }
        self.nzyxs = {
            i: etsynseg.pcdutil.reverse_coord(seg.results[f"nxyz{i}"])
            for i in self.labels
        }
    
    def calc_membrano(self, label):
        """ Calculate membranogram for one component.

        Args:
            label (int): Label of the component to be calculated.
        """
        # get points, normals
        zyx = np.copy(self.zyxs[label])
        nzyx = self.nzyxs[label]

        # clip points and tomo
        margin_nm = np.max(np.abs(self.dists)) * self.pixel_nm
        I, clip_low, _ = etsynseg.modutil.read_tomo_clip(
            self.tomo_file, zyx,
            margin_nm=margin_nm, pixel_nm=self.pixel_nm
        )
        zyx -= clip_low

        # calc membrano values
        v = etsynseg.membranogram.interpolate_distarr(
            zyx, nzyx, self.dists, I
        )
        v = np.mean(v, axis=0)

        # projection
        xyz = etsynseg.pcdutil.reverse_coord(zyx)
        nxyz = etsynseg.pcdutil.reverse_coord(nzyx)
        memproj = etsynseg.membranogram.Project().fit(xyz, nxyz)
        p1, p2 = memproj.transform(xyz)
        # align e1 towards +x direction
        e1 = memproj.e1
        if e1[0] < 0:
            e1 *= -1
            p1 *= -1
        # reshape to (npts,2)
        proj = np.transpose([p1, p2])

        # assemble results
        membno = dict(
            v=v,
            proj=proj,
            e1=e1
        )
        return membno

    def plot_membrano(self, membnos, qrange=(0.02, 0.98), cmap="gray", save=None):
        """ Plot membranograms.

        Args:
            membnos (dict): membnos[label] is the result of self.calc_membrano(label)
            qrange (2-tuple): Quantile range for indensity rescaling.
            cmap (str): Colormap.
            save (str): Filename for figure saving.
        """
        # point-array, convert to nm
        xy_arr = []
        for i in self.labels:
            xy_i = membnos[i]["proj"]*self.pixel_nm
            xy_arr.append(xy_i)

        # value-array, rescaled
        v_arr = []
        for i in self.labels:
            v_i = skimage.exposure.rescale_intensity(
                membnos[i]["v"],
                in_range=tuple(np.quantile(membnos[i]["v"], qrange))
            )
            v_arr.append(v_i)

        # plot
        fig, axes, _ = etsynseg.plot.scatter(
            xy_arr, v_arr=v_arr,
            cmap=cmap, shape=(len(self.labels), 1)
        )

        # plot labels
        # e1: angle of orientation in xy-plane 
        for idx, i in enumerate(self.labels):
            e1 = membnos[i]["e1"]
            e1_orient = np.rad2deg(np.arctan2(e1[1], e1[0]))
            axes[idx, 0].set(
                xlabel=rf"$component {i}, e_1(\theta_{{xy}}={e1_orient:.1f}^o)/nm$",
                ylabel=rf"$e_2(z)/nm$"
            )
        
        # saving
        if save is not None:
            fig.savefig(save)

    def workflow(self):
        """ Workflow of membranogram calculation and plotting.
        """
        # membranogram calculation
        membnos = {}
        for i in self.labels:
            membnos[i] = self.calc_membrano(i)
        
        # membranogram plotting
        self.plot_membrano(membnos, save=self.outputs+'.png')


if __name__ == "__main__":
    # init
    mem = SegMembrano()

    # read args
    args = vars(mem.argparser.parse_args())
    mem.load_args(args)

    # calc membranograms, plot and save
    mem.workflow()