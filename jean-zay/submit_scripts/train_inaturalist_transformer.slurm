#!/bin/bash
#SBATCH --job-name=TRI     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=30:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0-2

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


# batch_size[0]=32
# batch_size[1]=64
# batch_size[2]=128
# batch_size[3]=224
# batch_size[4]=384

loss[0]=naverap
autocast[0]=True

loss[1]=fastap
autocast[1]=True

loss[2]=blackboxap
autocast[2]=False


srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_deit' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_compa_deit' \
experience.max_iter=90 \
experience.num_workers=10 \
optimizer=inaturalist_deit \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=128 \
model=deit \
transform=inaturalist \
loss=${loss[${SLURM_ARRAY_TASK_ID}]} \
model.kwargs.with_autocast=${autocast[${SLURM_ARRAY_TASK_ID}]}
