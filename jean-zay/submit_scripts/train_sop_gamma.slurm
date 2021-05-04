#!/bin/bash
#SBATCH --job-name=gamma_test     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=gamma_test%A_%a.out # output file name
#SBATCH --error=gamma_test%A_%a.err  # error file name
#SBATCH --array=0-4

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

gamma[0]="0.001"
gamma[1]="0.005"
gamma[2]="0.01"
gamma[3]="0.015"
gamma[4]="0.02"


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_g${loss.0.kwargs.gamma}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/sop' \
experience.max_iter=200 \
dataset.sampler.kwargs.batch_size=128 \
loss=adaptativeap \
+loss.0.kwargs.gamma=${gamma[${SLURM_ARRAY_TASK_ID}]}
