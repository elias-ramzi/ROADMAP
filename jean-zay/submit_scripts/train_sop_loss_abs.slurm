#!/bin/bash
#SBATCH --job-name=TRLA     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=TRLA%A_%a.out # output file name
#SBATCH --error=TRLA%A_%a.err  # error file name
#SBATCH --array=0-1

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

loss[0]="smoothap_abs"
loss[1]="adaptativeap_abs"


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_deit' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop_contr' \
experience.max_iter=25 \
experience.num_workers=10 \
dataset.sampler.kwargs.batch_size=128 \
loss=contrastive
