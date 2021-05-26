#!/bin/bash
#SBATCH --job-name=GRID     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=8:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=%A_%a.out # output file name
#SBATCH --error=%A_%a.err  # error file name
#SBATCH --array=0-11

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


tau[0]=0.01
rho[0]=0.0
zeta[0]=0.05

tau[1]=0.01
rho[1]=0.1
zeta[1]=0.05

tau[2]=0.01
rho[2]=2.0
zeta[2]=0.05

tau[3]=0.01
rho[3]=5.0
zeta[3]=0.05

tau[4]=0.01
rho[4]=10.0
zeta[4]=0.05

tau[5]=0.01
rho[5]=100.0
zeta[5]=0.05


tau[6]=0.001
rho[6]=1.0
zeta[6]=0.005

tau[7]=0.05
rho[7]=1.0
zeta[7]=0.1

tau[8]=0.1
rho[8]=1.0
zeta[8]=0.1

tau[9]=1.0
rho[9]=1.0
zeta[9]=0.1



tau[10]=0.01
rho[10]=1000.0
zeta[10]=0.05

tau[11]=0.01
rho[11]=10000.0
zeta[11]=0.05


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_tau${loss.0.kwargs.temp}_rho${loss.0.kwargs.tau}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/margin_ap_grid' \
experience.max_iter=100 \
experience.num_workers=10 \
dataset.sampler.kwargs.batch_size=128 \
dataset.sampler.kwargs.batches_per_super_pair=10 \
loss=marginap_sigmo \
optimizer=sop_small \
loss.0.kwargs.tau=${rho[${SLURM_ARRAY_TASK_ID}]} \
loss.0.kwargs.temp=${tau[${SLURM_ARRAY_TASK_ID}]} \
loss.0.kwargs.margin=${zeta[${SLURM_ARRAY_TASK_ID}]}
