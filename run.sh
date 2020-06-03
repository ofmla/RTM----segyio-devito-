#!/bin/bash
export OMP_NUM_THREADS=5
export DEVITO_LANGUAGE=openmp
ulimit -s unlimited
module load anaconda3/5.2.0
source activate devito
echo 'This job started on: ' `date`
python RTM_BP_2007_repo.py >&1 | tee text.file

