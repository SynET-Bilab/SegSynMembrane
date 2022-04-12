__author__ = "Zhenghan Liao"
__version__ = "0.1"

# dependencies = [
## conda-forge
# "numpy", "scipy", "pandas",
# "scikit-learn", "scikit-image", "tslearn",
# "deap", "igraph", "leidenalg",
# "matplotlib", "napari",
# "mrcfile", "starfile", "h5py"
## pypi
# splipy
# open3d
# ]

# auxiliary functions
from etsynseg import utils, io, plot, tracing, bspline, membranogram
# membrane detection
from etsynseg import hessian, dtvoting, nonmaxsup
# membrane extraction
from etsynseg import dividing, evomsac, matching, meshrefine
# workflows
from etsynseg import workflows
