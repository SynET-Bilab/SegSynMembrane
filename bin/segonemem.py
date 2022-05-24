#!/usr/bin/env python
""" Segmentation of one membrane.
"""
import multiprocessing
import etsynseg

class SegOneMem(etsynseg.segbase.SegBase):
    def __init__(self):
        """ Initialization
        """
        # init from SegBase
        super().__init__(prog="segonemem", labels=(1,))
        
        # build parser
        self.build_argparser()
    
    def build_argparser(self):
        """ Amemd build_parser for segmentation.

        Mainly setting the default values.
        """
        super().build_argparser()

        self.argparser.set_defaults(
            # basics
            extend=20,
            neigh_thresh=5,
            # detect
            detect_smooth=5,
            detect_tv=20,
            detect_filt=2,
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

    def components_one(self):
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
            zyx1 = etsynseg.components.extract_components_one(
                zyx,
                r_thresh=tomod["neigh_thresh"],
                min_size=min_size
            )
        # RuntimeError: component size < min_size
        except RuntimeError as e:
            self.logger.error(e)
            raise

        # save results
        self.steps["components"]["zyx1"] = zyx1

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
            results[f"nxyz{i}"] = etsynseg.pcdutil.reverse_coord(nzyx_i)

            # surface area
            area_i, _ = etsynseg.moosac.surface_area(
                zyx_i, tomod["guide"],
                len_grid=tomod["neigh_thresh"]*4
            )
            results[f"area{i}_nm2"] = area_i * pixel_nm**2

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
            self.load_tomod(interp_degree=2, raise_noref=False)
        elif mode in ["runfine"]:
            self.load_tomod(interp_degree=1, raise_noref=False)

        # detecting
        self.detect()

        # extract components
        self.components_one()

        # fit refine
        self.fit_refine(1)

        # finalize
        self.final_results()
        self.final_outputs()
        self.logger.info(f"""segmentation finished: total {self.timer.total()}""")

if __name__ == "__main__":
    # init
    seg = SegOneMem()
    
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
    