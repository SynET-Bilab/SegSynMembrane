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
