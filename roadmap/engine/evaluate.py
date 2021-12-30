import os

import torch
from pytorch_metric_learning import testers
import pytorch_metric_learning.utils.common_functions as c_f
from tqdm import tqdm

import roadmap.utils as lib
from .accuracy_calculator import get_accuracy_calculator


class GlobalEmbeddingSpaceTester(testers.GlobalEmbeddingSpaceTester):

    def label_levels_to_evaluate(self, query_labels):
        num_levels_available = query_labels.shape[1]
        if self.label_hierarchy_level == "all":
            return range(num_levels_available)
        elif isinstance(self.label_hierarchy_level, int):
            assert self.label_hierarchy_level < num_levels_available
            return [self.label_hierarchy_level]
        elif c_f.is_list_or_tuple(self.label_hierarchy_level):
            # assert max(self.label_hierarchy_level) < num_levels_available
            return self.label_hierarchy_level

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        s, e = 0, 0
        with torch.no_grad():
            lib.LOGGER.info("Computing embeddings")
            # added the option of disabling TQDM
            for i, data in enumerate(tqdm(dataloader, disable=os.getenv('TQDM_DISABLE'))):
                img, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, "all", self.label_mapper)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(
                        len(dataloader.dataset),
                        label.size(1),
                        device=self.data_device,
                        dtype=label.dtype,
                    )
                    all_q = torch.zeros(
                        len(dataloader.dataset),
                        q.size(1),
                        device=self.data_device,
                        dtype=q.dtype,
                    )
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
        return all_q, labels


def get_tester(
    normalize_embeddings=False,
    batch_size=64,
    num_workers=16,
    pca=None,
    exclude_ranks=None,
    k=2047,
    **kwargs,
):
    calculator = get_accuracy_calculator(
        exclude_ranks=exclude_ranks,
        k=k,
        **kwargs,
    )

    return GlobalEmbeddingSpaceTester(
        normalize_embeddings=normalize_embeddings,
        data_and_label_getter=lambda batch: (batch["image"], batch["label"]),
        batch_size=batch_size,
        dataloader_num_workers=num_workers,
        accuracy_calculator=calculator,
        data_device=None,
        pca=pca,
    )


@lib.get_set_random_state
def evaluate(
    net,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    epoch=None,
    tester=None,
    custom_eval=None,
    **kwargs
):
    at_R = 0

    dataset_dict = {}
    splits_to_eval = []
    if train_dataset is not None:
        dataset_dict["train"] = train_dataset
        splits_to_eval.append(('train', ['train']))
        at_R = max(at_R, train_dataset.my_at_R)

    if val_dataset is not None:
        dataset_dict["val"] = val_dataset
        splits_to_eval.append(('val', ['val']))
        at_R = max(at_R, val_dataset.my_at_R)

    if test_dataset is not None:
        if isinstance(test_dataset, dict):
            if 'gallery' in test_dataset:
                dataset_dict.update(test_dataset)
                splits_to_eval.append(('test', ['gallery']))
                at_R = max(at_R, test_dataset['test'].my_at_R, test_dataset['gallery'].my_at_R)
            elif 'distractor' in test_dataset:
                dataset_dict.update(test_dataset)
                splits_to_eval.append(('test', ['test', 'distractor']))
                at_R = max(at_R, test_dataset['test'].my_at_R, test_dataset['distractor'].my_at_R)
        elif isinstance(test_dataset, list):
            for dts in test_dataset:
                dataset_dict.update(dts)
                names = list(dts.keys())
                at_R = max(at_R, list(dts.values())[0].my_at_R, list(dts.values())[1].my_at_R)
                splits_to_eval.append((
                    names[0] if names[0].startswith("query") else names[1],
                    [names[0] if names[0].startswith("gallery") else names[1]]
                ))
        else:
            dataset_dict["test"] = test_dataset
            splits_to_eval.append(('test', ['test']))
            at_R = max(at_R, test_dataset.my_at_R)

    if custom_eval is not None:
        dataset_dict = custom_eval["dataset"]
        splits_to_eval = custom_eval["splits"]

    if tester is None:
        # next lines usefull when computing only the mAP@R and small recall values
        # if ('k' not in kwargs) and (at_R != 0):
        #     kwargs["k"] = at_R + 1
        tester = get_tester(**kwargs)

    return tester.test(
        dataset_dict=dataset_dict,
        epoch=f"{epoch}",
        trunk_model=net,
        splits_to_eval=splits_to_eval,
    )
