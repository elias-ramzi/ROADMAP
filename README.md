# Robust And Decomposable Average Precision for Image Retrieval (NeurIPS 2021)

This repository contains the source code for our [ROADMAP paper (NeurIPS 2021)](https://arxiv.org/abs/2110.01445).

![outline](https://github.com/elias-ramzi/ROADMAP/blob/main/picture/outline.png)

## Use ROADMAP

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/robust-and-decomposable-average-precision-for/image-retrieval-on-inaturalist)](https://paperswithcode.com/sota/image-retrieval-on-inaturalist?p=robust-and-decomposable-average-precision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/robust-and-decomposable-average-precision-for/image-retrieval-on-sop)](https://paperswithcode.com/sota/image-retrieval-on-sop?p=robust-and-decomposable-average-precision-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/robust-and-decomposable-average-precision-for/image-retrieval-on-cub-200-2011)](https://paperswithcode.com/sota/image-retrieval-on-cub-200-2011?p=robust-and-decomposable-average-precision-for)

## Datasets

We use the following datasets for our submission

- CUB-200-2011 (download link available on this website : http://www.vision.caltech.edu/visipedia/CUB-200.html)
- Stanford Online Products (you can download it here : https://cvgl.stanford.edu/projects/lifted_struct/)
- INaturalist-2018 (obtained from here https://github.com/visipedia/inat_comp/tree/master/2018#Data)


## Run the code

<details>
  <summary><b>SOP</b></summary><br/>

  The following command reproduce our results for Table 4.

  ```
CUDA_VISIBLE_DEVICES=0 python roadmap/single_experiment_runner.py \
'experience.experiment_name=sop_ROADMAP_${dataset.sampler.kwargs.batch_size}_sota' \
experience.seed=333 \
experience.max_iter=100 \
'experience.log_dir=${env:HOME}experiments/ROADMAP' \
optimizer=sop \
model=resnet \
transform=sop_big \
dataset=sop \
dataset.sampler.kwargs.batch_size=128 \
dataset.sampler.kwargs.batches_per_super_pair=10 \
loss=roadmap
  ```

  With the transformer backbone :

  ```
  CUDA_VISIBLE_DEVICES=0 python roadmap/single_experiment_runner.py \
  'experience.experiment_name=sop_ROADMAP_${dataset.sampler.kwargs.batch_size}_DeiT' \
  experience.seed=333 \
  experience.max_iter=75 \
  'experience.log_dir=${env:HOME}/experiments/ROADMAP' \
  optimizer=sop_deit \
  model=deit \
  transform=sop \
  dataset=sop \
  dataset.sampler.kwargs.batch_size=128 \
  dataset.sampler.kwargs.batches_per_super_pair=10 \
  loss=roadmap
  ```
</details>


<details>
  <summary><b>INaturalist</b></summary><br/>

  For ROADMAP sota results:

  ```
CUDA_VISIBLE_DEVICES='0,1,2' python roadmap/single_experiment_runner.py \
'experience.experiment_name=inat_ROADMAP_${dataset.sampler.kwargs.batch_size}_sota' \
experience.seed=333 \
experience.max_iter=90 \
'experience.log_dir=experiments/ROADMAP' \
optimizer=inaturalist \
model=resnet \
transform=inaturalist \
dataset=inaturalist \
dataset.sampler.kwargs.batch_size=384 \
loss=roadmap_inat
  ```
</details>


<details>
  <summary><b>CUB-200-2011</b></summary><br/>

  For ROADMAP sota results:

  ```
  CUDA_VISIBLE_DEVICES=0 python roadmap/single_experiment_runner.py \
  'experience.experiment_name=cub_ROADMAP_${dataset.sampler.kwargs.batch_size}_sota' \
  experience.seed=333 \
  experience.max_iter=200 \
  'experience.log_dir=${env:HOME}/experiments/ROADMAP' \
  optimizer=cub \
  model=resnet_max_ln \
  transform=cub_big \
  dataset=cub \
  dataset.sampler.kwargs.batch_size=128 \
  loss=roadmap
  ```

  ```
  CUDA_VISIBLE_DEVICES=0 python roadmap/single_experiment_runner.py \
  'experience.experiment_name=cub_ROADMAP_${dataset.sampler.kwargs.batch_size}_sota_DeiT' \
  experience.seed=333 \
  experience.max_iter=150 \
  'experience.log_dir=${env:HOME}/experiments/ROADMAP' \
  optimizer=cub_deit \
  model=deit \
  transform=cub \
  dataset=cub \
  dataset.sampler.kwargs.batch_size=128 \
  loss=roadmap
  ```

</details>


The results are not exactly the same as my code changed a bit (for instance the random seed are not the same).


## Contacts

If you have any questions don't hesitate to create an issue on this repository. Or send me an email at elias.ramzi@lecnam.net.

Don't hesitate to cite our work:
```
@inproceedings{
ramzi2021robust,
title={Robust and Decomposable Average Precision for Image Retrieval},
author={Elias Ramzi and Nicolas THOME and Cl{\'e}ment Rambour and Nicolas Audebert and Xavier Bitot},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=VjQw3v3FpJx}
}
```


## Resources
- Pytorch Metric Learning (PML): https://github.com/KevinMusgrave/pytorch-metric-learning
- SmoothAP: https://github.com/Andrew-Brown1/Smooth_AP
- Blackbox: https://github.com/martius-lab/blackbox-backprop
- FastAP: https://github.com/kunhe/FastAP-metric-learning
- SoftBinAP: https://github.com/naver/deep-image-retrieval
- timm: https://github.com/rwightman/pytorch-image-models
- PyTorch: https://github.com/pytorch/pytorch
- Hydra: https://github.com/facebookresearch/hydra
- Faiss: https://github.com/facebookresearch/faiss
- Ray: https://github.com/ray-project/ray
