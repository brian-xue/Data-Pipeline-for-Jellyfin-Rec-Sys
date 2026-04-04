from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_weighted_user_embedding(movie_embeddings: Iterable[list[float]], weights: Iterable[float]) -> list[float] | None:
    vectors = [np.asarray(emb, dtype=np.float32) for emb in movie_embeddings]
    ws = [float(weight) for weight in weights]

    if not vectors or not ws or len(vectors) != len(ws):
        return None

    dims = {vec.shape for vec in vectors}
    if len(dims) != 1:
        return None

    weight_sum = float(sum(ws))
    if weight_sum <= 0:
        return None

    stacked = np.stack(vectors, axis=0)
    weight_array = np.asarray(ws, dtype=np.float32).reshape(-1, 1)
    fused = (stacked * weight_array).sum(axis=0) / weight_sum
    return fused.astype(np.float32).tolist()
