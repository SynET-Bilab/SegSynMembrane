__author__ = "Zhenghan Liao"
__version__ = "0.1"

dependencies = [
    "numpy", "numba", "scikit-learn", "scikit-image", "mrcfile"
]

from synseg import io, imgprocess
from synseg import dtvoting, nonmaxsup, cluster
