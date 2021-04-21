#!/bin/bash
#SBATCH --job-name=dwnlddata     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3         # le calcul va etre long
#SBATCH --partition=archive
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $STORE

curl -C - "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz" --output train_val2018.tar.gz
curl -C - "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz" --output train2018.json.tar.gz
curl -C - "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz" --output val2018.json.tar.gz
