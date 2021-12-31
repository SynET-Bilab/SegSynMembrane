#!/usr/bin/env python
""" plot: plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import napari
import plotly
import plotly.subplots

__all__ = [
    # matplotlib: 2d plot
    "imshow", "scatter",
    # napari: 3d viewer
    "imshow3d",
    # plotly: tooltips
    "imshowly"
]


#=========================
# matplotlib
#=========================

def setup_subplots(n, shape, figsize1):
    """ setup subplots
    :param n: number of subplots, can be different from shape
    :param shape: (nrows, ncols), either can be None
    :param figsize1: size of one subplot
    :return: fig, axes
    """
    # setup shape
    if shape is None:
        shape = (1, n)
    elif (shape[0] is None) and (shape[1] is None):
        shape = (1, n)
    elif shape[0] is None:  # (None, ncols)
        shape = (int(np.ceil(n/shape[1])), shape[1])
    elif shape[1] is None:  # (nrows, None)
        shape = (shape[0], int(np.ceil(n/shape[0])))

    # setup figure
    figsize = (
        figsize1[0]*shape[1],  # size_x * ncols
        figsize1[1]*shape[0]  # size_y * nrows
    )
    fig, axes = plt.subplots(
        nrows=shape[0], ncols=shape[1],
        sharex=True, sharey=True,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False  # always return axes as 2d array
    )
    return fig, axes

def imshow(
        I_arr, shape=None, style="binary",
        vrange=None, qrange=(0, 1),
        cmap="gray", colorbar=True, colorbar_shrink=0.6,
        title_arr=None, suptitle=None,
        supxlabel=None, supylabel=None,
        figsize1=(3, 3), save=None
    ):
    """ show multiple images
    :param I_arr: a 1d list of images
    :param shape: (nrows, ncols), will auto set if either is None
    :param style: custom, gray, binary, orient
    :param vrange, qrange: range of value(v) or quantile(q)
    :param cmap, colorbar, colorbar_shrink: set colors
    :param title_arr, suptitle, supxlabel, supylabel: set labels
    :param figsize1: size of one subplot
    :param save: name to save fig
    :return: fig, axes
    """
    # setup styles
    # grayscale image: adjust qrange
    if style == "gray":
        cmap = "gray"
        vrange = None
        qrange = (0.02, 0.98)
    # orientation: circular cmap, convert rad to deg
    elif style == "orient":
        cmap = "hsv"
        vrange = (0, 180)
        I_arr = [np.mod(I, np.pi)/np.pi*180 for I in I_arr]
    elif style == "binary":
        cmap = "gray"
        vrange = (0, 0.1)
        colorbar = False
    elif style == "custom":
        pass
    else:
        raise ValueError("style options: custom, gray, orient")

    # setup subplots
    fig, axes = setup_subplots(len(I_arr), shape, figsize1)
    shape = axes.shape

    # plot on each ax
    for idx1d, I in enumerate(I_arr):
        # get 2d index
        idx2d = np.unravel_index(idx1d, shape)

        # setup vrange
        if vrange is not None:
            vmin, vmax = vrange
        else:
            vmin = np.quantile(I, qrange[0])
            vmax = np.quantile(I, qrange[1])
        
        # plot
        axes[idx2d].set_aspect(1)
        im = axes[idx2d].imshow(
            I, vmin=vmin, vmax=vmax,
            cmap=cmap, origin="lower"
        )

        # setup colorbar, title
        if colorbar:
            fig.colorbar(im, ax=axes[idx2d], shrink=colorbar_shrink)
        if title_arr is not None:
            axes[idx2d].set_title(title_arr[idx1d])
        
    # setup fig title
    fig.suptitle(suptitle)
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)

    # save fig
    if save is not None:
        fig.savefig(save)

    return fig, axes

def scatter(
        xy_arr,
        labels_arr=None,
        shape=None,
        marker_size=0.1,
        cmap="viridis", colorbar=True, colorbar_shrink=0.6,
        title_arr=None, suptitle=None,
        supxlabel="x/pixel", supylabel="y/pixel",
        figsize1=(3.5, 3.5), save=None
    ):
    """ show multiple scatters
    :param xy_arr: 1d list of 2d array [x, y]
    :param labels_arr: 1d list of labels for each xy
    :param shape: (nrows, ncols), will auto set if either is None
    :param cmap, colorbar, colorbar_shrink: set colors
    :param title_arr, suptitle, supxlabel, supylabel: set labels
    :param figsize1: size of one subplot
    :param save: name to save fig
    :return: fig, axes
    """
    # regularize labels_arr
    if labels_arr is None:
        labels_arr = [None]*len(xy_arr)

    # setup subplots
    fig, axes = setup_subplots(len(xy_arr), shape, figsize1)
    shape = axes.shape

    # plot on each ax
    for idx1d, (xy, labels) in enumerate(zip(xy_arr, labels_arr)):
        # get 2d index
        idx2d = np.unravel_index(idx1d, shape)

        # plot
        axes[idx2d].set_aspect(1)
        im = axes[idx2d].scatter(
            xy[:, 0], xy[:, 1],
            s=marker_size,
            c=labels, cmap=cmap
        )

        # setup colorbar, title
        if colorbar:
            fig.colorbar(im, ax=axes[idx2d], shrink=colorbar_shrink)
        if title_arr is not None:
            axes[idx2d].set_title(title_arr[idx1d])

    # setup fig title
    fig.suptitle(suptitle)
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)

    # save fig
    if save is not None:
        fig.savefig(save)

    return fig, axes


#=========================
# napari
#=========================

def imshow3d(
        I_back, I_fronts=None,
        opacity_back=0.75, opacity_front=1,
        cmap_back="gray", cmap_fronts="default"
    ):
    """ overlay images using napari, through different channels
    :param I_back: single background image
    :param I_fronts: array of front images
    :param isin_01: is I already in range 01, otherwise minmax scale
    :param opacity_back, opacity_front: setup opacity
    :param cmap_back: colormap
    :param cmap_fronts: default=["green", "yellow", "cyan", "magenta",
        "bop blue", "bop orange", "bop purple", "red", "blue"]
    """
    # setup cmap, opacity according to number of front images
    if I_fronts is None:
        I_fronts = []
    n_fronts = len(I_fronts)
    if cmap_fronts == "default":
        cmap_fronts = [
            "green", "yellow", "cyan", "magenta",
            "bop blue", "bop orange", "bop purple", "red", "blue"
        ]
        cmap_fronts = cmap_fronts[:n_fronts]
    opacity_fronts = [opacity_front]*n_fronts
    name_fronts = [f"foreground {i}" for i in range(1, n_fronts+1)]

    # flip y-axis, napari doesn't seem to support orient="lower" as in imshow
    I_back = np.flip(I_back, -2)
    I_fronts = [np.flip(I, -2) for I in I_fronts]

    # stack images along axis-0
    I_stack = np.stack([I_back, *I_fronts], axis=0)

    # view images via channels
    viewer = napari.view_image(
        I_stack, channel_axis=0,
        name=["background", *name_fronts],
        colormap=[cmap_back, *cmap_fronts],
        opacity=[opacity_back, *opacity_fronts]
    )
    return viewer


#=========================
# plotly
#=========================

def imshowly(I_arr, cmap=None, renderers="vscode"):
    """ 2d imshow using plotly (more interactive)
    :param I_arr: a 1d list of images
    :param cmap: set colorscale
    """
    # setup
    plotly.io.renderers.default = renderers
    n = len(I_arr)
    fig = plotly.subplots.make_subplots(rows=1, cols=n)
    fig.update_layout(yaxis=dict(scaleanchor='x'))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    
    # plot
    for i in range(n):
        fig.add_trace(
            plotly.graph_objects.Heatmap(
                z=I_arr[i], colorscale=cmap
            ),
            row=1, col=i+1  # 1-based
        )
    return fig
