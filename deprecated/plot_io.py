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

# def segs_to_model(seg_arr, model_name, voxel_size):
#     """ convert segmentations to model
#     :param seg_arr: [seg1,seg2,...], binary images
#     :param model_name: name for output models, without .mod
#     :param voxel_size: (x,y,z), or None for auto
#     :return: None
#         outputs model file model_name.mod
#     """
#     # for each seg, convert to mrc, then to mod
#     mod_name_arr = []
#     for i, seg in enumerate(seg_arr):
#         # write seg to a temp mrc
#         mrc = tempfile.NamedTemporaryFile(suffix=".mrc")
#         mrc_name = mrc.name
#         write_mrc(seg, mrc_name, voxel_size=voxel_size)

#         # convert mrc to mod
#         # imodauto options
#         # -h find contours around pixels higher than this value
#         # -n find inside contours in closed, annular regions
#         # -m minimum are for each contour
#         mod_name_i = f"{model_name}_{i}.mod"
#         subprocess.run(
#             f"imodauto -h 1 -n -m 1 {mrc_name} {mod_name_i}",
#             shell=True, check=True
#         )
#         mod_name_arr.append(mod_name_i)
#         mrc.close()

#     # imodjoin options:
#     # -c change colors of objects being copied to first model
#     mod_name_str = " ".join(mod_name_arr)
#     mod_name_joined = f"{model_name}.mod"
#     subprocess.run(
#         f"imodjoin -c {mod_name_str} {mod_name_joined}",
#         shell=True, check=True
#     )
