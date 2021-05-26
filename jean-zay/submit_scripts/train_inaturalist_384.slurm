#!/bin/bash
#SBATCH --job-name=64     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%j.out # output file name
#SBATCH --error=%j.err  # error file name

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models

srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_rho100' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_compa_64' \
experience.max_iter=100 \
optimizer=inaturalist \
experience.num_workers=10 \
transform=inaturalist \
dataset=inaturalist \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
dataset.sampler.kwargs.batch_size=64 \
loss=marginap_sigmo_abs \
+loss.0.kwargs.start=0.0 \
loss.0.kwargs.offset=1.0 \
loss.0.kwargs.tau=100.0
