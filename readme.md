# readme

## steps for using on siat-icbi hpc

step 0: update module path (needed for the first time)
open `~/.bashrc`, add `/share/root/user/liaozh/modulefiles` to `$MODULEPATH`

step 1: generate model
open tomo_file in 3dmod
create object 1: boundary for segmentation (width can be 2~3 cleft width)
create object 2: a point in the presynaptic region
save as model_file

step 2: segmentation
run commands in terminal:
```
module load etsynseg
segprepost_run.py tomo_file model_file --voxel_size <vx>
```
model_file can be omitted if it has the same name as tomo_file, with suffix replaced by `.mod`
voxel_size can be omitted if it is contained in the header of tomo_file

outputs: (1 for presynapse, 2 for postsynapse)
name-steps.npz: info of all steps
name-seg.npz: point cloud (xyz1,2) and normals (normal1,2)
name-seg.png: image of a few slices
name-seg.mod: an imod model file for viewing results

misc:
rerun some steps with new parameters
`segprepost_run.py name-steps.npz --<option> <param>`
diagnosis (based on napari, may not work well on hpc)
`segprepost_run.py name-steps.npz -d`
help
`segprepost_run.py -h`


## diagnosis for not-so-good segmentations

if too many interfering lines around the membrane: refine model
in the model, make the boundary closer to the membranes.

if two membranes are wrongly divided: amend model
in the model, create object 3 with a dividing line in the cleft.

if missing too many lines: loosen filters
set `--detect_zfilter` to a small number (e.g. 10), which filters out connected components whose span in z is smaller than this number.

if the membrane is far from reaching the boundary in the xy direction: shrink side grids
set `--evomsac_shrinkside` to a smaller number (e.g. 0.1), which makes the side grids for EvoMSAC sampling this ratio times the normal grids.