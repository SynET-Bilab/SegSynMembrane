__author__ = "Zhenghan Liao"
__version__ = "0.1"

# dependencies = [
# "numpy", "scipy", "pandas",
# "scikit-learn", "scikit-image",
# "deap",
# "matplotlib", "napari",
# "mrcfile",
# "splipy"
# ]

from synseg import utils, io, plot, trace, bspline # auxiliary
from synseg import hessian, dtvoting, nonmaxsup   # membrane detection
from synseg import division, evomsac, matching  #W membrane extraction
from synseg import workflow  # integrated workflows
