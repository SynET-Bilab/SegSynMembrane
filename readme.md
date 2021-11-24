# readme

[TOC]

## intro

- For pre-/post-synaptic membrane segmentation.

## notes on the code

- parallel computing
  - dask: works well, by distributing data via disk
    - tried distributing by passing data, workers got killed for unknown reasons
  - numba: does not work with fft
  - multiprocessing: cannot pickle tv2d

## issues

- dtvoting.tv3d_dask
  - port issues when not starting client from `__main__` ([github issue](https://github.com/dask/distributed/issues/726))
  - generated tempfile npy's, which may not be removed if execution is terminated halfway
