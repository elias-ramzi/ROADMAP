#!/bin/bash
#SBATCH --job-name=large_emb     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=large_emb%j.out # output file name
#SBATCH --error=large_emb%j.err  # error file name

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.7.1

export TORCH_HOME=${SCRATCH}/pretrained_models


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_2048_nolinear${model.kwargs.without_fc}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
experience.resume=rolling.ckpt \
'experience.log_dir=${env:SCRATCH}/experiments/sop' \
experience.max_iter=200 \
experience.force_lr=0.000001 \
dataset.sampler.kwargs.batch_size=128 \
model.kwargs.embed_dim=2048 \
loss=adaptativeap
