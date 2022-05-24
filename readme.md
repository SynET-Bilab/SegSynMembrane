# readme

## steps for using etsynseg
(on siat-icbi hpc)

step 0: update module path (needed for the first time)
edit `~/.bashrc`, add `/share/root/user/liaozh/modulefiles` to `$MODULEPATH`

step 1: generate model
open tomo_file in 3dmod
create object 1: guiding line along the synaptic cleft
create object 2: a point in the presynaptic region
save as model_file

alternatives: run `segdrawmodel.sh *.mrc` to draw on tomos one by one.

step 2: segmentation
run commands in terminal:
```
module load etsynseg
segprepost.py run tomo_file model_file -o outputs -px <pixel_size_nm> --extend <extend_width_nm>
```
model file: can be omitted if it has the same name as tomo_file but with suffix replaced by '.mod'
pixel size: can be omitted if it is contained in the header of tomo_file
outputs: can be omitted, then will be set to model_file with '.mod' replaced by '-seg'

alternatives: run `segsbatchtemplate.sh` to generate a sbatch file for job submission.

outputs: (1 for presynapse, 2 for postsynapse)
name-seg.npz: info of all steps and results.
    results can be retrieved in python using: `np.load(name-seg.npz, allow_pickle=True)["results"].item()`
name-seg.png: image of sample slices
name-seg.mod: an imod model file with segmented membranes and the manual bounding region

visualization:
run `segview.py args/steps/3d/moosac state.npz`

## tools

segdrawmodel.sh: open mrc one by one for drawing models.
segsbatchtemplate.sh: generate a template sbatch file.
segonemem.py: segmemtation of one membrane.
segview.py: visualize segmentation results.

## diagnosis for unsatisfying segmentations

if two membranes are wrongly divided: refine model
draw the guiding lines such that they separate pre from post.
run with "segprepost.py runfine tomo_file model_file"
