#!/bin/bash
#SBATCH --job-name=train_inat     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=30:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=train_inat%A_%a.out # output file name
#SBATCH --error=train_inat%A_%a.err  # error file name
#SBATCH --array=0-1

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

typerank[0]="True"
typerank[1]="False"

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_${model.kwargs.backbone_name}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat' \
experience.max_iter=90 \
experience.num_workers=10 \
optimizer=inaturalist \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=224 \
transform=inaturalist \
model.kwargs.with_autocast=True \
loss=marginap \
+loss.0.kwargs.with_true_rank=${typerank[${SLURM_ARRAY_TASK_ID}]}
