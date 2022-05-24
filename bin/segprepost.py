#!/usr/bin/env python
""" Segmentation of pre- and post-synapses.
"""

import multiprocessing
import numpy as np
import etsynseg

class SegPrePost(etsynseg.segbase.SegBase):
    def __init__(self):
        """ Initialization
        """
        # init from SegBase
        super().__init__(prog="segprepost", labels=(1, 2))
        
        # update fields
        self.results.update({f"dist{i}_nm": None for i in (1, 2)})

        # build parser
        self.build_argparser()

    def build_argparser(self):
        """ Amemd build_parser for segmentation.

        Mainly setting the default values.
        """
        super().build_argparser()

        self.argparser.set_defaults(
            # basics
            extend=40,
            neigh_thresh=5,
            # detect
            detect_smooth=5,
            detect_tv=0,
            detect_filt=4,
            detect_supp=0.5,
            # components
            components_min=0.5,
            # moosac
            moosac_lengrids=[50, 150],
            moosac_shrinkside=0.25,
            moosac_resize=5,
            moosac_popsize=40,
            moosac_tol=0.005,
            moosac_maxiter=150,
        )

    def components_auto(self):
        """ Extract two components automatically.

        Prerequisites: membranes are detected.
        Effects: updates self.steps["components"].
        """
        # log
        self.timer.click()

        # setup
        args = self.args
        tomod = self.steps["tomod"]
        
        # extract by division
        zyx = self.steps["detect"]["zyx"]
        min_size = len(tomod["guide"])*args["components_min"]
        try:
            zyx1, zyx2 = etsynseg.components.extract_components_two(
                zyx,
                r_thresh=tomod["neigh_thresh"],
                orients=None, sigma_dO=np.pi/4,
                min_size=min_size
            )
        # RuntimeError: component size < min_size
        except RuntimeError as e:
            self.logger.error(e)
            raise

        # sort by distance to ref point
        zyx1, zyx2 = etsynseg.pcdutil.sort_pcds_by_ref(
            [zyx1, zyx2],
            pt_ref=tomod["normal_ref"]
        )
        
        # save results
        self.steps["components"]["zyx1"] = zyx1
        self.steps["components"]["zyx2"] = zyx2

        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""extracted components: {self.timer.click()}""")

    def components_by_mask(self):
        """ Extract two components using masks.
        
        Prerequisites: membranes are detected.
        Effects: updates self.steps["components"].
        """
        # log
        self.timer.click()

        # setup
        tomod = self.steps["tomod"]

        # extract by masking
        zyx = self.steps["detect"]["zyx"]
        min_size = len(tomod["guide"])*args["components_min"]
        # pre in normal minus, post in normal plus
        try:
            zyx1, zyx2 = etsynseg.components.extract_components_regions(
                zyx,
                region_arr=[tomod["bound_minus"], tomod["bound_plus"]],
                r_thresh=tomod["neigh_thresh"],
                min_size=min_size
            )
        # RuntimeError: component size < min_size
        except RuntimeError as e:
            self.logger.error(e)
            raise

        # save results
        self.steps["components"]["zyx1"] = zyx1
        self.steps["components"]["zyx2"] = zyx2

        # log
        self.save_state(self.args["outputs_state"])
        self.logger.info(f"""extracted components: {self.timer.click()}""")

    def final_results(self):
        """ Calculate final results. Save in self.results.
        """
        # log
        self.timer.click()

        # setup
        tomod = self.steps["tomod"]
        meshrefine = self.steps["meshrefine"]
        pixel_nm = tomod["pixel_nm"]
        zyx_low = tomod["clip_low"]

        # collect results
        results = {}
        for i in self.labels:
            # points
            zyx_i = meshrefine[f"zyx{i}"]
            results[f"xyz{i}"] = etsynseg.pcdutil.reverse_coord(zyx_low+zyx_i)

            # normals
            nzyx_i = etsynseg.pcdutil.normals_points(
                zyx_i,
                sigma=tomod["neigh_thresh"]*2,
                pt_ref=tomod["normal_ref"]
            )
            if i == 2:  # flip sign for postsynapse
                nzyx_i = -nzyx_i
            results[f"nxyz{i}"] = etsynseg.pcdutil.reverse_coord(nzyx_i)

            # surface area
            area_i, _ = etsynseg.moosac.surface_area(
                zyx_i, tomod["guide"],
                len_grid=tomod["neigh_thresh"]*4
            )
            results[f"area{i}_nm2"] = area_i * pixel_nm**2

        # distance
        dist1, dist2 = etsynseg.pcdutil.points_distance(
            meshrefine["zyx1"], meshrefine["zyx2"],
            return_2to1=True
        )
        results["dist1_nm"] = dist1 * pixel_nm
        results["dist2_nm"] = dist2 * pixel_nm

        # save
        self.results.update(results)
        self.save_state(self.args["outputs_state"])

        # log
        self.logger.info(f"finalized: {self.timer.click()}")

    def final_outputs(self):
        """ Final outputs.
        """
        self.output_model(self.args["outputs"]+".mod")
        self.output_slices(self.args["outputs"]+".png", nslice=5)

    def workflow(self):
        """ Segmentation workflow.
        """
        mode = self.args["mode"]

        # load tomod
        if mode in ["run"]:
            self.load_tomod(interp_degree=2, raise_noref=True)
        elif mode in ["runfine"]:
            self.load_tomod(interp_degree=1, raise_noref=True)

        # detecting
        self.detect()

        # extract components
        if mode in ["runfine"]:
            self.components_by_mask()
        else:
            self.components_auto()

        # fit refine
        self.fit_refine(1)
        self.fit_refine(2)

        # finalize
        self.final_results()
        self.final_outputs()
        self.logger.info(f"""segmentation finished: total {self.timer.total()}""")

if __name__ == "__main__":
    # init
    seg = SegPrePost()
    
    # parse and load args
    args = vars(seg.argparser.parse_args())
    seg.load_args(args)
    
    # workflow
    # read mode from args, not seg.args
    mode = args["mode"]
    # run segmentation
    if mode in ["run", "runfine"]:
        pool = multiprocessing.Pool()
        seg.register_map(pool.map)
        seg.workflow()
        seg.register_map()
        pool.close()
    # recalculate and continue calculating final results
    elif mode == "contresults":
        seg.final_results()
        seg.final_outputs()
