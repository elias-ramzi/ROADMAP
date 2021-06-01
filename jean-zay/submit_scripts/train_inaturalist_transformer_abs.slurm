#!/bin/bash
#SBATCH --job-name=I384     # job name
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
#SBATCH --array=0-5

set -x
cd $WORK/margin_ap

module purge
module load pytorch-gpu/py3/1.8.0

export TORCH_HOME=${SCRATCH}/pretrained_models


batch_size[0]=128
loss[0]=marginap_sigmo_inat
name[0]='experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}'

batch_size[1]=128
loss[1]=marginap_sigmo_abs_inat
name[1]='experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}'


batch_size[2]=224
loss[2]=marginap_sigmo_inat
name[2]='experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}'

batch_size[3]=224
loss[3]=marginap_sigmo_abs_inat
name[3]='experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}'


batch_size[4]=384
loss[4]=marginap_sigmo_inat
name[4]='experience.experiment_name=inat_${loss.0.name}_${dataset.sampler.kwargs.batch_size}'

batch_size[5]=384
loss[5]=marginap_sigmo_abs_inat
name[5]='experience.experiment_name=inat_${loss.0.name}_${loss.1.name}_${dataset.sampler.kwargs.batch_size}'


sleep $(( 60*${SLURM_ARRAY_TASK_ID}))
srun python margin_ap/single_experiment_runner.py \
${name[${SLURM_ARRAY_TASK_ID}]} \
'dataset.kwargs.data_dir=${env:SCRATCH}/Inaturalist' \
experience.seed=333 \
'experience.log_dir=${env:SCRATCH}/experiments/inat_ablation_study_deit' \
experience.max_iter=90 \
experience.num_workers=10 \
experience.save_model=50 \
optimizer=inaturalist_deit \
model=deit \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
transform=inaturalist \
loss=${loss[${SLURM_ARRAY_TASK_ID}]} \
