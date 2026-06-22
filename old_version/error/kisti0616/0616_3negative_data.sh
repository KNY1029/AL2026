#!/bin/sh
#PBS -N 0616_3negative_data
#PBS -q normal
#PBS -A etc
#PBS -l select=4:ncpus=64:mpiprocs=64
#PBS -l walltime=47:59:59
#PBS -m abe
#PBS -M nahappy1029@gmail.com

cd /scratch/a2051a01

# 환경 변수 충돌 방지용 안전장치 (BASH_FUNC_module 에러 해결)
unset -f module
unset -f BASH_FUNC_module%%

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate algorithm

module purge
module load intel/19.1.2 impi/19.1.2

# 워커 노드로 함수 전달되는 것을 막기 위해 env로 초기화 후 mpirun 실행
env -u BASH_FUNC_module%% mpirun python negative_data_kisti2.py
