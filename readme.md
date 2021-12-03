# readme

[TOC]

## intro

- For pre-/post-synaptic membrane segmentation.

## usage

```python
# read, negate
I = synseg.imgprocess.read_mrc("I.mrc")
I = synseg.imgprocess.negate(I)

# features
S, O = synseg.imgprocess.features3d_hessian(I, sigma=5)

# mask
# TODO: add converting point to mask

# TV
S_tv, O_tv = synseg.dtvoting.tv3d(S, O, sigma=10)

# NMS
nms = synseg.nonmaxsup.nms3d(S, O, sigma=5)

```

## notes on the code

- parallel computing
  - numba: multithreading, but simpler and faster than dask
    - the issue of incompatible fft can be resolved with objmode
  - dask: works well, by distributing data via disk
    - tried distributing by passing data, workers got killed for unknown reasons
    - although multiprocessing, behaves slower than numba on typical data size
  - multiprocessing: cannot pickle tv2d
