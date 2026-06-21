#!/bin/sh
#PBS -N ZINC_Download
#PBS -V
#PBS -q normal
#PBS -A etc
#PBS -l select=1:ncpus=4:mpiprocs=4
#PBS -l walltime=48:00:00

# 작업 시작, 종료, 에러 시 이메일 알림
#PBS -m abe
#PBS -M nahappy1029@gmail.com

# 스크래치 작업 디렉터리로 이동
cd /scratch/a2051a01

# Conda 환경 활성화 (pandas 등 사용을 위해 필수)
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate algorithm

# 병렬(MPI) 다운로드가 아니므로 mpirun 없이 파이썬으로 바로 실행합니다.
python download_zinc.py
