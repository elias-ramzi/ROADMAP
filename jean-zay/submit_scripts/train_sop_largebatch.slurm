#!/bin/bash
#SBATCH --job-name=XBM     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=30:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=%j.out # output file name
#SBATCH --error=%j.err  # error file name

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_xbm${memory.kwargs.size}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop_xbm' \
experience.max_iter=100 \
optimizer=sop_small \
experience.num_workers=10 \
dataset.sampler.kwargs.batch_size=128 \
memory=sop \
memory.kwargs.unique=False \
memory.kwargs.size=2048 \
loss=marginap_sigmo_abs
