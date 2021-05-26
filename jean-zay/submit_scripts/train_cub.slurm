#!/bin/bash
#SBATCH --job-name=MARGIN     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0-7

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


pos_margin[0]=0.9
neg_margin[0]=0.8

pos_margin[1]=0.9
neg_margin[1]=0.7

pos_margin[2]=0.9
neg_margin[2]=0.5

pos_margin[3]=0.9
neg_margin[3]=0.4

pos_margin[4]=0.9
neg_margin[4]=0.3

pos_margin[5]=0.9
neg_margin[5]=0.2

pos_margin[6]=0.9
neg_margin[6]=0.1

pos_margin[7]=0.9
neg_margin[7]=0.0


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_neg${loss.1.kwargs.neg_margin}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_grid_margin' \
experience.max_iter=90 \
experience.num_workers=10 \
optimizer=inaturalist \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=224 \
transform=inaturalist \
loss=marginap_sigmo_abs \
+loss.0.kwargs.start=0.0 \
loss.0.kwargs.offset=1.0 \
loss.1.kwargs.pos_margin=${pos_margin[${SLURM_ARRAY_TASK_ID}]} \
loss.1.kwargs.neg_margin=${neg_margin[${SLURM_ARRAY_TASK_ID}]}
