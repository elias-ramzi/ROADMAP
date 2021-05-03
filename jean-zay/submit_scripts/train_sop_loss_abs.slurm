#!/bin/bash
#SBATCH --job-name=train_sop     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=train_sop%A_%a.out # output file name
#SBATCH --error=train_sop%A_%a.err  # error file name
#SBATCH --array=0-1

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.7.1

export TORCH_HOME=${SCRATCH}/pretrained_models

# loss[0]="smoothap_abs"
# loss[1]="smoothap_abs"
loss[0]="adaptativeap_abs"
loss[1]="adaptativeap_abs"

batch_size[0]="32"
batch_size[1]="128"
# batch_size[2]="32"
# batch_size[3]="128"


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop' \
experience.max_iter=200 \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
loss=${loss[${SLURM_ARRAY_TASK_ID}]} \
loss.1.weight=1.0
