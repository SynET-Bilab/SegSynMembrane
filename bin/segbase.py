""" 
"""

import time
import numpy as np
# from etsynseg import matching, meshrefine, nonmaxsup
# from etsynseg import imgutil, pcdutil, modutil, io
# from etsynseg import features, dtvoting
# from etsynseg import moosac

__all__ = [
    "Timer", "SegBase"
]

class Timer:
    """ A timer class.

    Examples:
        timer = Timer()
        dt = timer.click()
    """
    def __init__(self):
        """ Init and record current time.
        """
        self.t_last = time.perf_counter()

    def click(self):
        """ Record current time and calc time difference.
        """
        t_curr = time.perf_counter()
        del_t = t_curr - self.t_last
        self.t_last = t_curr
        del_t = f"{del_t:.1f}s"
        return del_t

class SegBase:
    """ Base class for segmentation.
    """
    def __init__(self):
        self.args = {}
        self.steps = {}
        self.results = {}

    def view_status(self):
        """ view status (finished, process_time) of each step
        """
        status = {
            k: {"finished": v["finished"], "process_time": v["timing"]}
            for k, v in self.steps.items()
        }
        return status

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

    def save_state(self, state_file):
        """ Save data to state file.

        Args:
            state_file (str): Filename of the state file.
        """
        np.savez_compressed(
            state_file,
            args=self.args,
            steps=self.steps,
            results=self.results
        )

    def check_steps(self, steps_prev, raise_error=False):
        """ raise error if any prerequisite steps is not finished
            steps_prev: array of names of prerequisite steps
            raise_error: if raise error when prerequisites are not met
        """
        satisfied = True
        for step in steps_prev:
            if not self.steps[step]["finished"]:
                if raise_error:
                    raise RuntimeError(f"unsatisfied prerequisite step: {step}")
                satisfied = False
        return satisfied

