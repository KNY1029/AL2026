
#!/bin/sh
#PBS -N 0615_negative_data2
#PBS -V
#PBS -q normal
#PBS -A etc
#PBS -l select=4:ncpus=64:mpiprocs=64
#PBS -l walltime=47:59:59


# 작업 시작, 종료, 에러 시 이메일 알림
#PBS -m abe
#PBS -M nahappy1029@gmail.com

cd /scratch/a2051a01

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate algorithm

module purge
module load intel/19.1.2 impi/19.1.2

mpirun python negative_data_kisti2.py



