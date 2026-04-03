import os
import io
import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import faiss


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# Config
# =========================

@dataclass
class BuildIndexConfig:
    s3_uri: str
    output_dir: str = "/mnt/block/embedding/"
    embedding_column: str = "embedding"
    id_column: Optional[str] = "movie_id"

    # faiss options
    index_type: str = "flatip"   # flatip / flatl2
    normalize: bool = True       # usually True for cosine-style retrieval

    # s3 client options
    aws_region: Optional[str] = "eu-central-1"
    aws_endpoint_url: Optional[str] = "http://minio:9000"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


# =========================
# S3 Helpers
# =========================

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    path = s3_uri[len("s3://"):]
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def build_s3_client(cfg: BuildIndexConfig):
    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=cfg.aws_region,
        endpoint_url=cfg.aws_endpoint_url,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
    )
    return client


def read_s3_bytes(s3_client, s3_uri: str) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)
    logger.info("Downloading from S3: %s", s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


# =========================
# Data Loading
# =========================

def _convert_embedding_series_to_numpy(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series whose each row is a list / np.ndarray embedding
    into a 2D float32 numpy array.
    """
    values = series.tolist()
    if len(values) == 0:
        raise ValueError("Embedding column is empty.")

    first = values[0]
    if isinstance(first, np.ndarray):
        arr = np.vstack(values)
    else:
        arr = np.asarray(values)

    if arr.ndim != 2:
        raise ValueError(f"Embedding array must be 2D, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_from_parquet(
    s3_client,
    s3_uri: str,
    embedding_column: str,
    id_column: Optional[str],
):
    raw = read_s3_bytes(s3_client, s3_uri)
    df = pd.read_parquet(io.BytesIO(raw))

    print(f"Columns: {list(df.columns)}")

    if embedding_column not in df.columns:
        raise KeyError(f"Missing column: {embedding_column}")

    logger.info("Columns in parquet: %s", list(df.columns))

    if embedding_column not in df.columns:
        raise KeyError(
            f"Embedding column '{embedding_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if id_column and id_column not in df.columns:
        raise KeyError(
            f"ID column '{id_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    emb_series = df[embedding_column]

    first = emb_series.iloc[0]

    if isinstance(first, (list, tuple, np.ndarray)):
        embeddings = np.stack(emb_series.to_numpy())
    else:
        raise ValueError(
            f"Unsupported embedding format: {type(first)}. "
            f"Expected list/array per row."
        )

    if embeddings.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got shape={embeddings.shape}")

    embeddings = embeddings.astype(np.float32, copy=False)

    logger.info("Embedding shape: %s", embeddings.shape)

    ids = None
    if id_column and id_column in df.columns:
        ids = df[id_column].to_numpy()

        # 确保长度一致
        if len(ids) != len(embeddings):
            raise ValueError("IDs and embeddings size mismatch")

    return embeddings, ids

def load_from_npy(
    s3_client,
    s3_uri: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    raw = read_s3_bytes(s3_client, s3_uri)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)

    if arr.ndim != 2:
        raise ValueError(f"NPY embedding array must be 2D, got shape={arr.shape}")

    return arr.astype(np.float32, copy=False), None


def load_embeddings(
    cfg: BuildIndexConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    s3_client = build_s3_client(cfg)

    if cfg.s3_uri.endswith(".parquet"):
        embeddings, ids = load_from_parquet(
            s3_client=s3_client,
            s3_uri=cfg.s3_uri,
            embedding_column=cfg.embedding_column,
            id_column=cfg.id_column,
        )
    elif cfg.s3_uri.endswith(".npy"):
        embeddings, ids = load_from_npy(
            s3_client=s3_client,
            s3_uri=cfg.s3_uri,
        )
    else:
        raise ValueError(
            f"Unsupported input format for {cfg.s3_uri}. "
            f"Only .parquet and .npy are supported."
        )

    logger.info("Loaded embeddings shape: %s", embeddings.shape)
    if ids is not None:
        logger.info("Loaded ids shape: %s", ids.shape)

    return embeddings, ids


# =========================
# Faiss
# =========================

def maybe_normalize(embeddings: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return embeddings

    embeddings = embeddings.astype(np.float32, copy=False)
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flatip",
) -> faiss.Index:
    dim = embeddings.shape[1]
    index_type = index_type.lower()

    if index_type == "flatip":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "flatl2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(
            f"Unsupported index_type={index_type}. "
            f"Only flatip and flatl2 are supported for now."
        )

    index.add(embeddings)
    logger.info("Faiss index built, ntotal=%d, dim=%d", index.ntotal, dim)
    return index


# =========================
# Save
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_artifacts(
    output_dir: str,
    embeddings: np.ndarray,
    ids: Optional[np.ndarray],
    index: faiss.Index,
    meta: dict,
):
    ensure_dir(output_dir)

    embedding_npy_path = os.path.join(output_dir, "embedding.npy")
    index_path = os.path.join(output_dir, "faiss.index")
    meta_path = os.path.join(output_dir, "meta.json")

    logger.info("Saving embeddings to %s", embedding_npy_path)
    np.save(embedding_npy_path, embeddings)

    if ids is not None:
        ids_path = os.path.join(output_dir, "ids.npy")
        logger.info("Saving ids to %s", ids_path)
        np.save(ids_path, ids)

    logger.info("Saving faiss index to %s", index_path)
    faiss.write_index(index, index_path)

    logger.info("Saving meta to %s", meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================

def build_and_save_index(cfg: BuildIndexConfig):
    embeddings, ids = load_embeddings(cfg)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    embeddings = maybe_normalize(embeddings, cfg.normalize)
    index = build_faiss_index(embeddings, cfg.index_type)

    meta = {
        "source_s3_uri": cfg.s3_uri,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "index_type": cfg.index_type,
        "normalize": cfg.normalize,
        "embedding_column": cfg.embedding_column,
        "id_column": cfg.id_column,
        "has_ids": ids is not None,
    }

    save_artifacts(
        output_dir=cfg.output_dir,
        embeddings=embeddings,
        ids=ids,
        index=index,
        meta=meta,
    )

    logger.info("All artifacts saved to %s", cfg.output_dir)


if __name__ == "__main__":
    cfg = BuildIndexConfig(
        s3_uri="s3://embedding/movie_embeddings.parquet",
        output_dir="/mnt/block/embedding/",
        embedding_column="embedding",
        aws_access_key_id = "minioadmin",
        aws_secret_access_key = "minioadmin123",
        id_column="movieId",
        index_type="flatip",
        normalize=True,
    )

    build_and_save_index(cfg)