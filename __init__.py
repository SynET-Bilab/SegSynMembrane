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
from .utilities import imgutil, pcdutil
from .utilities import io, plot
from .utilities import bspline
from .utilities import modutil

from .submodules import features, nonmaxsup, dtvoting
from .submodules import detecting, components
from .submodules import moosac, matching, meshrefine

# workflows
# from etsynseg import workflows

from .submodules import membranogram