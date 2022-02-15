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
segprepost_script.py tomo_file model_file --voxel_size <vx>
```
