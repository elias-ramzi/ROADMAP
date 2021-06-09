# Robust And Decomposable Average Precision for Image Retrieval

![outline](https://github.com/elias-ramzi/ROADMAP/blob/clean_repo/picture/outline.png)

## Use ROADMAP

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```


## Datasets

We use the following datasets for our submission

- CUB-200-2011 (download link available on this website : http://www.vision.caltech.edu/visipedia/CUB-200.html)
- Stanford Online Products (you can download it here : https://cvgl.stanford.edu/projects/lifted_struct/)
- INaturalist-2018 (obtained from here https://github.com/visipedia/inat_comp/tree/master/2018#Data)


## Run the code

Here we give an example on how to reproduce the result of Table 2. of the paper (SOP, BS=64).

```
CUDA_VISIBLE_DEVICES=0 python roadmap/single_experiment_runner.py \
'experience.experiment_name=sop_${loss.0.name}_${dataset.sampler.kwargs.batch_size}' \
experience.seed=333 \
experience.max_iter=100 \
'experience.log_dir=${env:HOME}/experiments/ROADMAP' \
optimizer=sop \
model=resnet \
transform=sop \
dataset=sop \
dataset.sampler.kwargs.batch_size=64 \
dataset.sampler.kwargs.batches_per_super_pair=20 \
loss=roadmap
```
