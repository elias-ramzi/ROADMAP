#!/bin/bash
#SBATCH --job-name=train_inaturalist_384     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:2              # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=train_sop%j.out # output file name
#SBATCH --error=train_sop%j.ouy  # error file name

set -x
cd $WORK/Smooth_AP

module purge
module load pytorch-gpu/py3/1.7.1

python main.py \
--fc_lr_mul 2 \
--bs 384 \
--dataset inaturalist \
--source_path ${SCRATCH} \
--save_path ${WORK}/inat_smoothap
