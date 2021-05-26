#!/bin/bash
#SBATCH --job-name=ABS     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=25:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0-6

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


weight0[0]=1.0
weight1[0]=1.0

weight0[1]=1.8
weight1[1]=0.2

weight0[2]=0.2
weight1[2]=1.8

weight0[3]=0.6
weight1[3]=1.4

weight0[4]=1.4
weight1[4]=0.6

weight0[5]=1.6
weight1[5]=0.4

weight0[6]=0.4
weight1[6]=1.6


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}_lambda${loss.1.weight}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_lambda' \
experience.max_iter=100 \
experience.num_workers=10 \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=224 \
optimizer=inaturalist \
loss=marginap_sigmo_abs \
loss.0.weight=${weight0[${SLURM_ARRAY_TASK_ID}]} \
loss.1.weight=${weight1[${SLURM_ARRAY_TASK_ID}]} \
+loss.0.kwargs.start=0.0 \
loss.0.kwargs.offset=1.0
