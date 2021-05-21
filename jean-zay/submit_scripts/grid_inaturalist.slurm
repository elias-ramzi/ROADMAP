#!/bin/bash
#SBATCH --job-name=GRID     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0-3

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

weight[0]=0.1
weight[1]=2
weight[2]=5
weight[3]=10

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_lambda${loss.1.weight}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_grid_lambda' \
experience.max_iter=90 \
experience.num_workers=10 \
optimizer=inaturalist \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=384 \
transform=inaturalist \
loss=marginap_sigmo_abs \
+loss.0.kwargs.start=0.0 \
loss.0.kwargs.offset=1.0 \
loss.1.weight=${weight[${SLURM_ARRAY_TASK_ID}]}
