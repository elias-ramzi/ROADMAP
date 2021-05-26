#!/bin/bash
#SBATCH --job-name=IXBM     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=50:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


batch_size[0]=32
batch_size[1]=64
batch_size[2]=128
batch_size[3]=224
batch_size[4]=384


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_full' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_true_xbm' \
experience.max_iter=90 \
experience.num_workers=10 \
optimizer=inaturalist \
dataset=inaturalist \
memory=inaturalist \
dataset.sampler.kwargs.batch_size=64 \
transform=inaturalist \
loss=pair_loss \
memory.kwargs.unique=False \
memory.kwargs.size= \
experience.test_eval_freq=-1
