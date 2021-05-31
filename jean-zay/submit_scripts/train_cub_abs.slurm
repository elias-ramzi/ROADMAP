#!/bin/bash
#SBATCH --job-name=TRIERR     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=30:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=%A_%a_tierr.out # output file name
#SBATCH --error=%A_%a_tierr.err  # error file name
#SBATCH --array=0-20

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


loss[0]=blackboxap
seed[0]=0
with_autocast[0]=False

loss[1]=blackboxap
seed[1]=1
with_autocast[1]=False

loss[2]=blackboxap
seed[2]=2
with_autocast[2]=False

loss[3]=blackboxap
seed[3]=3
with_autocast[3]=False


loss[4]=fastap
seed[4]=0
with_autocast[4]=True

loss[5]=fastap
seed[5]=1
with_autocast[5]=True

loss[6]=fastap
seed[6]=2
with_autocast[6]=True

loss[7]=fastap
seed[7]=3
with_autocast[7]=True


loss[8]=naverap
seed[8]=0
with_autocast[8]=True

loss[9]=naverap
seed[9]=1
with_autocast[9]=True

loss[10]=naverap
seed[10]=2
with_autocast[10]=True

loss[11]=naverap
seed[11]=3
with_autocast[11]=True


loss[12]=marginap_sigmo_abs_inat
seed[12]=0
with_autocast[12]=True

loss[13]=marginap_sigmo_abs_inat
seed[13]=1
with_autocast[13]=True

loss[14]=marginap_sigmo_abs_inat
seed[14]=2
with_autocast[14]=True

loss[15]=marginap_sigmo_abs_inat
seed[15]=3
with_autocast[15]=True


loss[16]=smoothap
seed[16]=0
with_autocast[16]=True

loss[17]=smoothap
seed[17]=1
with_autocast[17]=True

loss[18]=smoothap
seed[18]=2
with_autocast[18]=True

loss[19]=smoothap
seed[19]=3
with_autocast[19]=True

loss[20]=marginap_sigmo_abs_inat
seed[20]=333
with_autocast[20]=True


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
'experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_seed${experience.seed}' \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=${seed[${SLURM_ARRAY_TASK_ID}]} \
'experience.log_dir=${env:SCRATCH}/experiments/inat_ranking_compa_deit' \
experience.max_iter=90 \
experience.num_workers=10 \
experience.save_model=50 \
model.kwargs.with_autocast=${with_autocast[${SLURM_ARRAY_TASK_ID}]} \
optimizer=inaturalist_deit \
model=deit \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=128 \
transform=inaturalist \
loss=${loss[${SLURM_ARRAY_TASK_ID}]}
