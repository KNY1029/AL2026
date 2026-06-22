#!/bin/sh
#PBS -N 0620_negative_data
#PBS -V
#PBS -q normal
#PBS -A etc
# 노드당 16개 코어만 사용하여 프로세스당 약 6GB 메모리 확보
#PBS -l select=4:ncpus=64:mpiprocs=16
#PBS -l walltime=19:59:59
#PBS -m abe
#PBS -M nahappy1029@gmail.com

cd /scratch/a2051a01

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate algorithm

module purge
module load intel/19.1.2 impi/19.1.2

unset -f module

mpirun python negative_data_kisti4.py
