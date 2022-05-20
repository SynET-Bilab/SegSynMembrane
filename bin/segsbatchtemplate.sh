#!/usr/bin/env bash

echo "----segsbatchtemplate----"
echo "Generate template.sbatch"

fsbatch="template.sbatch"

if [[ -f ${fsbatch} ]]
then
    mv ${fsbatch} ${fsbatch}~
fi

cat > ${fsbatch} <<EOL
#!/bin/bash
#SBATCH --partition=tao
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --job-name=segmem
#SBATCH --output=slurm-%j.out

source /usr/share/Modules/init/bash

module purge
module load etsynseg 

# segprepost.py run tomo.mrc model.mod -o outputs -px pixel_size_nm

# for fmrc in *.mrc
# do
#     segprepost.py run $fmrc
# done
EOL