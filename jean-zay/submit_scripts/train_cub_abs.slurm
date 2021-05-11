#!/bin/bash
#SBATCH --job-name=cub_abs     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=05:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=cub_abs%A_%a.out # output file name
#SBATCH --error=cub_abs%A_%a.err  # error file name
#SBATCH --array=0-11

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


loss[0]="smoothap_abs"
loss[1]="smoothap_abs"
loss[2]="smoothap_abs"
loss[3]="smoothap_abs"
loss[4]="adaptativeap_abs"
loss[5]="adaptativeap_abs"
loss[6]="adaptativeap_abs"
loss[7]="adaptativeap_abs"
loss[8]="marginap_abs"
loss[9]="marginap_abs"
loss[10]="marginap_abs"
loss[11]="marginap_abs"

batch_size[0]="32"
batch_size[1]="64"
batch_size[2]="128"
batch_size[3]="224"
batch_size[4]="32"
batch_size[5]="64"
batch_size[6]="128"
batch_size[7]="224"
batch_size[8]="32"
batch_size[9]="64"
batch_size[10]="128"
batch_size[11]="224"

srun python margin_ap/single_experiment_runner.py \
--multirun \
'experience.experiment_name=cub_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_${model.kwargs.backbone_name}' \
experience.seed=300 \
'experience.log_dir=${env:SCRATCH}/experiments/cub_final' \
experience.max_iter=201 \
experience.test_eval_freq=10 \
experience.num_workers=10 \
experience.save_model=10 \
optimizer=cub \
optimizer.0.kwargs.lr=0.000001 \
transform=cub \
dataset=cub \
'dataset.kwargs.data_dir=${env:SCRATCH}/CUB_200_2011/' \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
