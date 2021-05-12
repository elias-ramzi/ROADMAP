import logging

import numpy as np
import torch
import faiss
import pytorch_metric_learning.utils.common_functions as c_f


def get_knn(references, queries, num_k, embeddings_come_from_same_source):
    # device = queries.device
    d = references.size(-1)
    references = c_f.to_numpy(references).astype(np.float32)
    queries = c_f.to_numpy(queries).astype(np.float32)

    if embeddings_come_from_same_source:
        num_k += 1

    logging.info("running k-nn with k=%d" % num_k)
    logging.info("embedding dimensionality is %d" % d)

    index = faiss.IndexFlatL2(d)
    if torch.cuda.device_count() > 1:
        co = faiss.GpuMultipleClonerOptions()
        co.shards = True
        index = faiss.index_cpu_to_all_gpus(index, co)
    else:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(references)
    distances, indices = index.search(queries, num_k)
    distances = c_f.to_device(torch.from_numpy(distances), device='cpu')
    indices = c_f.to_device(torch.from_numpy(indices), device='cpu')
    index.reset()
    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]
    return indices, distances
