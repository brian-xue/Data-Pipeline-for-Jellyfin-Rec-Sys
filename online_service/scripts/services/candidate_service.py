from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


def _storage_options(config) -> dict[str, Any] | None:
    if config is None:
        return None
    storage = getattr(config, "object_storage", None)
    if storage is None:
        return None
    options: dict[str, Any] = {
        "client_kwargs": {"endpoint_url": storage.endpoint},
    }
    if storage.access_key:
        options["key"] = storage.access_key
    if storage.secret_key:
        options["secret"] = storage.secret_key
    return options

import ast
import json

def _normalize_embedding_value(value):
    if value is None:
        raise ValueError("user embedding is None")

    # if list like
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]

    # if string
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("user embedding is empty string")

        try:
            parsed = json.loads(s)
        except Exception:
            parsed = ast.literal_eval(s)

        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"user embedding string did not parse to list: {type(parsed)}")

        return [float(v) for v in parsed]

    raise ValueError(f"unsupported embedding value type: {type(value)}")


def _read_user_embedding(user_id: int, user_embedding_uri: str, config) -> list[float] | None:
    read_kwargs = {}
    storage_options = _storage_options(config)
    if storage_options is not None and user_embedding_uri.startswith("s3://"):
        read_kwargs["storage_options"] = storage_options

    if user_embedding_uri.endswith(".json"):
        import json

        if user_embedding_uri.startswith("s3://"):
            import s3fs

            storage = config.object_storage if config is not None else None
            fs = s3fs.S3FileSystem(
                key=getattr(storage, "access_key", None),
                secret=getattr(storage, "secret_key", None),
                client_kwargs={"endpoint_url": getattr(storage, "endpoint", None)} if storage else None,
            )
            with fs.open(user_embedding_uri.replace("s3://", "", 1), "rb") as handle:
                payload = json.load(handle)
        else:
            with open(user_embedding_uri, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        if int(payload.get("user_id", user_id)) != user_id:
            return None
        embedding = payload.get("embedding")
        return None if embedding is None else _normalize_embedding_value(embedding)

    frame = pd.read_parquet(user_embedding_uri, **read_kwargs)
    if "user_id" in frame.columns:
        matched = frame[frame["user_id"] == user_id]
        if matched.empty:
            return None
        row = matched.iloc[0]
    else:
        if frame.empty:
            return None
        row = frame.iloc[0]

    for column in ("user_embedding", "embedding", "long_term_embedding"):
        if column in frame.columns:
            value = row[column]
            if value is not None:
                return _normalize_embedding_value(value)
    return None


def _load_movie_embeddings(config) -> tuple[np.ndarray, np.ndarray | None]:
    candidate_cfg = config.candidate
    embeddings = np.load(candidate_cfg.movie_embeddings_npy_path).astype(np.float32)
    movie_ids = None
    if candidate_cfg.movie_ids_path and os.path.exists(candidate_cfg.movie_ids_path):
        movie_ids = np.load(candidate_cfg.movie_ids_path)
    return embeddings, movie_ids


def _load_or_build_index(movie_embeddings: np.ndarray, config):
    candidate_cfg = config.candidate
    if faiss is not None and candidate_cfg.movie_embedding_index_path and os.path.exists(candidate_cfg.movie_embedding_index_path):
        index = faiss.read_index(candidate_cfg.movie_embedding_index_path)
        return index

    if faiss is None:
        return None

    normalized = movie_embeddings.astype(np.float32).copy()
    faiss.normalize_L2(normalized)
    index = faiss.IndexHNSWFlat(normalized.shape[1], int(candidate_cfg.hnsw_m))
    index.hnsw.efSearch = int(candidate_cfg.hnsw_ef_search)
    index.add(normalized)
    return index


def retrieve_by_embedding(user_id: int, user_embedding_uri: str | None, top_k: int, config) -> list[dict[str, Any]]:
    if not user_embedding_uri:
        return []

    user_embedding = _read_user_embedding(user_id=user_id, user_embedding_uri=user_embedding_uri, config=config)
    if not user_embedding:
        return []

    movie_embeddings, movie_ids = _load_movie_embeddings(config)
    if movie_embeddings.size == 0:
        return []

    query = np.asarray(user_embedding, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != movie_embeddings.shape[1]:
        return []

    index = _load_or_build_index(movie_embeddings, config)
    if index is None:
        normalized_movies = movie_embeddings.copy()
        normalized_query = query.copy()
        normalized_movies /= np.linalg.norm(normalized_movies, axis=1, keepdims=True) + 1e-12
        normalized_query /= np.linalg.norm(normalized_query, axis=1, keepdims=True) + 1e-12
        scores = normalized_movies @ normalized_query[0]
        order = np.argsort(-scores)[:top_k]
        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(order, start=1):
            movie_id = int(movie_ids[idx]) if movie_ids is not None else int(idx)
            movie_embedding = movie_embeddings[idx]
            results.append({"movie_id": movie_id, "score": float(scores[idx]), "rank": rank, "embedding": movie_embedding.tolist()})
        return results, user_embedding

    if faiss is not None:
        normalized_query = query.copy()
        faiss.normalize_L2(normalized_query)
        search_k = min(top_k, movie_embeddings.shape[0])
        distances, indices = index.search(normalized_query, search_k)
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0:
                continue
            movie_id = int(movie_ids[idx]) if movie_ids is not None else int(idx)
            score = 1.0 - float(distance) / 2.0
            movie_embedding = movie_embeddings[idx]
            results.append({"movie_id": movie_id, "score": score,  "rank": rank, "embedding": movie_embedding.tolist()})
        return results, user_embedding

    return []


def retrieve_from_popular(popular_movies: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in popular_movies[:top_k]:
        items.append(
            {
                "movie_id": int(row["movie_id"]),
                "score": float(row.get("score", 0.0)),
            }
        )
    return items


def merge_candidates(
    embedding_candidates: list[dict[str, Any]],
    popular_candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()

    for item in embedding_candidates + popular_candidates:
        movie_id = int(item["movie_id"])
        if movie_id in seen:
            continue
        seen.add(movie_id)
        merged.append(item)
        if len(merged) >= top_k:
            break

    for index, item in enumerate(merged, start=1):
        item["rank"] = index

    return merged
