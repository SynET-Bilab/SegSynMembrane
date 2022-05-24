""" Utils: misc utilities.
"""
import time

__all__ = [
    "Timer"
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
        """ Record current time and return elapsed time since the last click.

        Returns:
            del_t (str or float): Elapsed time.
        """
        t_curr = time.perf_counter()
        del_t = t_curr - self.t_last
        self.t_last = t_curr
        if self.return_format == "string":
            del_t = f"{del_t:.1f}s"
        return del_t

    def total(self):
        """ Return elapsed time since initiation.

        Returns:
            del_t (str or float): Elapsed time.
        """
        t_curr = time.perf_counter()
        del_t = t_curr - self.t_init
        if self.return_format == "string":
            del_t = f"{del_t:.1f}s"
        return del_t
