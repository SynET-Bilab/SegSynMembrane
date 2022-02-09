# import plotly
# import plotly.subplots

# def imshowly(I_arr, cmap=None, renderers="vscode"):
#     """ 2d imshow using plotly (more interactive)
#     :param I_arr: a 1d list of images
#     :param cmap: set colorscale
#     """
#     # setup
#     plotly.io.renderers.default = renderers
#     n = len(I_arr)
#     fig = plotly.subplots.make_subplots(rows=1, cols=n)
#     fig.update_layout(yaxis=dict(scaleanchor='x'))
#     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    
#     # plot
#     for i in range(n):
#         fig.add_trace(
#             plotly.graph_objects.Heatmap(
#                 z=I_arr[i], colorscale=cmap
#             ),
#             row=1, col=i+1  # 1-based
#         )
#     return fig
