#!/bin/sh
#PBS -N 0617_4negative_data
#PBS -V
#PBS -q normal
#PBS -A etc
#PBS -l select=4:ncpus=64:mpiprocs=64
#PBS -l walltime=47:59:59
#PBS -m abe
#PBS -M nahappy1029@gmail.com

cd /scratch/a2051a01

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate algorithm

# 1. MPI 실행에 필요한 모듈을 '먼저' 정상적으로 불러옵니다.
module purge
module load intel/19.1.2 impi/19.1.2

# 2. 모듈 로딩이 끝난 후, 워커 노드를 죽이는 주범인 module 함수만 메모리에서 안전하게 지웁니다.
unset -f module

# 3. 병렬 연산 실행 (이제 워커 노드들이 에러 없이 정상 작동합니다)
mpirun python negative_data_kisti3.py
