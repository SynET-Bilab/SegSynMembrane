__author__ = "Zhenghan Liao"
__version__ = "0.2"

# dependencies = [
## conda-forge
# "numpy", "scipy", "pandas",
# "scikit-learn", "scikit-image", "tslearn",
# "deap", "igraph",
# "matplotlib", "napari",
# "mrcfile", "starfile", "h5py"
## pypi
# splipy
# open3d > 0.15
# ]

# auxiliary functions
from .utilities import imgutils, pcdutils
from .utilities import io, plot
from .utilities import bspline
from .utilities import modutils

from .submodules import features, nonmaxsup, dtvoting
from .submodules import tracing, dividing
from .submodules import detecting, evomsac, matching, meshrefine

# workflows
# from etsynseg import workflows

from .submodules import membranogram