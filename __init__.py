__author__ = "Zhenghan Liao"
__version__ = "1.3.4"

# dependencies = [
## conda-forge
# "numpy", "scipy">1.6, "pandas",
# "scikit-learn", "scikit-image", "tslearn",
# "deap", "igraph",
# "matplotlib", "napari",
# "mrcfile", "starfile", "h5py"
## pypi
# splipy
# open3d > 0.15
# ]

# utilities
from .utilities import imgutil, pcdutil, miscutil
from .utilities import io, plot
from .utilities import bspline
from .utilities import modutil

# submodules - segmentation
from .submodules import features, nonmaxsup, dtvoting
from .submodules import detecting, components
from .submodules import moosac, matching, meshrefine

# submodules - auxiliary
from .submodules import membranogram, memsampling

# executable scripts
from .bin import segbase
from .bin import segprepost, segonemem
from .bin import segmembrano, segsampling
