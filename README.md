# MarginAP



 ## SOP

 srun python margin_ap/single_experiment_runner.py \
 'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}_schedule' \
 'dataset.kwargs.data_dir=${env:SCRATCH}/Stanford_Online_Products' \
 experience.seed=333 \
 'experience.log_dir=${env:SCRATCH}/experiments/sop' \
 experience.max_iter=300 \
 experience.resume=epoch_100.ckpt \
 'optimizer.0.scheduler_on_epoch=null' \
 'optimizer.1.scheduler_on_epoch=null' \
 experience.force_lr=0.000001 \
 dataset.sampler.kwargs.batch_size=${batch_size[${SLURM_ARRAY_TASK_ID}]} \
 loss=${loss[${SLURM_ARRAY_TASK_ID}]}
