#!/bin/bash
module purge
module load

export SOP_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
export TMP_DIR=$JOBSCRATCH/
export TRAIN_DIR=experiments
