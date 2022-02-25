# readme


- usage on siat-icbi hpc
```
# step 0
# update modulepath (needed for the first time)
# ~/.bashrc or ~/.bash_modules
# append /share/root/user/liaozh/modulefiles to $MODULEPATH

# step 1
# generate model
3dmod tomo_file
# create object 1: boundary for segmentation (width can be 2~3 cleft width)
# create object 2: a point in the presynaptic region
# save as model_file

# step 2
# segmentation
module load etsynseg
segprepost_run.py tomo_mrc model_mod --voxel_size <vx>

# outputs
# 1 - presynapse, 2 - postsynapse
# name-steps.npz: info of all steps
# name-seg.npz: point cloud (xyz1,2) and normals (normal1,2)
# name-seg.png: image of a few slices
# name-seg.mod: an imod model file for viewing results

# misc
# rerun some steps with new parameters
segprepost_run.py steps_npz --option <param>

# 
```
