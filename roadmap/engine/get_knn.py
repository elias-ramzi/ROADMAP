import numpy as np
import torch
import faiss
import pytorch_metric_learning.utils.common_functions as c_f

import roadmap.utils as lib


def get_knn(references, queries, num_k, embeddings_come_from_same_source, with_faiss=True):
    num_k += embeddings_come_from_same_source

    lib.LOGGER.info("running k-nn with k=%d" % num_k)
    lib.LOGGER.info("embedding dimensionality is %d" % references.size(-1))

    if with_faiss:
        distances, indices = get_knn_faiss(references, queries, num_k)
    else:
        distances, indices = get_knn_torch(references, queries, num_k)

    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]

    return indices, distances


def get_knn_faiss(references, queries, num_k):
    lib.LOGGER.debug("Computing k-nn with faiss")

    d = references.size(-1)
    device = references.device
    references = c_f.to_numpy(references).astype(np.float32)
    queries = c_f.to_numpy(queries).astype(np.float32)

    index = faiss.IndexFlatL2(d)
    try:
        if torch.cuda.device_count() > 1:
            co = faiss.GpuMultipleClonerOptions()
            co.shards = True
            index = faiss.index_cpu_to_all_gpus(index, co)
        else:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    except AttributeError:
        # Only faiss CPU is installed
        pass

    index.add(references)
    distances, indices = index.search(queries, num_k)
    distances = c_f.to_device(torch.from_numpy(distances), device=device)
    indices = c_f.to_device(torch.from_numpy(indices), device=device)
    index.reset()
    return distances, indices


def get_knn_torch(references, queries, num_k):
    lib.LOGGER.debug("Computing k-nn with torch")

    scores = queries @ references.t()
    distances, indices = torch.topk(scores, num_k)
    return distances, indices
