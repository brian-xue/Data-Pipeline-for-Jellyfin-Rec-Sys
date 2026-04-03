from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

from minio_s3 import (
    path_exists,
    resolve_input_path,
    s3_filesystem,
    to_s3_uri,
    upload_file_to_path,
)


# ============================================================
# Paths
# ============================================================


INPUT_PARQUET = "s3://cleaned/movie_embedding_text.parquet"
OUTPUT_DIR = "s3://embedding"
OUTPUT_PARQUET = f"{OUTPUT_DIR}/movie_embeddings.parquet"

BATCH_SIZE = 256


# ============================================================
# Field weights
# Higher weight = stronger contribution to final movie vector
# ============================================================
FIELD_WEIGHTS: dict[str, float] = {
    "overview": 0.25,
    "keywords_list": 0.20,
    "genres_list": 0.30,
    "top_user_tags": 0.10,
    "tagline": 0.05,
    "title": 0.04,
    "original_title": 0.02,
    "production_companies_list": 0.015,
    "production_countries_list": 0.01,
    "spoken_languages_list": 0.01,
    "original_language": 0.003,
    "adult": 0.001,
    "release_year": 0.05,
}

# Metadata to keep in output
OUTPUT_METADATA_COLUMNS = [
    "movieId",
    "imdbId",
    "tmdbId",
    "title",
    "original_title",
    "release_year",
    "adult",
    "original_language",
    "genres_list",
    "keywords_list",
    "top_user_tags",
]


# ============================================================
# Helpers
# ============================================================
def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return True
    return False


def list_to_text(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (list, tuple)):
        items = [clean_text(x) for x in value]
        items = [x for x in items if x]
        return ", ".join(items)
    return clean_text(value)


def format_field_text(field_name: str, value: Any) -> str:
    """
    Create a natural text snippet for each field.
    We embed each field separately, so this label is helpful but lightweight.
    """
    if is_missing(value):
        return ""

    if field_name == "overview":
        text = clean_text(value)
        return f"Overview: {text}" if text else ""

    if field_name == "keywords_list":
        text = list_to_text(value)
        return f"Keywords: {text}" if text else ""

    if field_name == "genres_list":
        text = list_to_text(value)
        return f"Genres: {text}" if text else ""

    if field_name == "top_user_tags":
        text = list_to_text(value)
        return f"User tags: {text}" if text else ""

    if field_name == "tagline":
        text = clean_text(value)
        return f"Tagline: {text}" if text else ""

    if field_name == "title":
        text = clean_text(value)
        return f"Title: {text}" if text else ""

    if field_name == "original_title":
        text = clean_text(value)
        return f"Original title: {text}" if text else ""

    if field_name == "production_companies_list":
        text = list_to_text(value)
        return f"Production companies: {text}" if text else ""

    if field_name == "production_countries_list":
        text = list_to_text(value)
        return f"Production countries: {text}" if text else ""

    if field_name == "spoken_languages_list":
        text = list_to_text(value)
        return f"Spoken languages: {text}" if text else ""

    if field_name == "original_language":
        text = clean_text(value)
        return f"Original language: {text}" if text else ""

    if field_name == "adult":
        text = clean_text(value)
        return f"Adult: {text}" if text else ""

    if field_name == "release_year":
        text = clean_text(value)
        return f"Release year: {text}" if text else ""

    text = list_to_text(value)
    return text


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """
    Encode only non-empty texts. Empty texts get zero vectors.
    """
    dim = model.get_sentence_embedding_dimension()
    output = np.zeros((len(texts), dim), dtype=np.float32)

    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    if not non_empty_indices:
        return output

    non_empty_texts = [texts[i] for i in non_empty_indices]

    encoded = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine-friendly field vectors
        show_progress_bar=False,
    ).astype(np.float32)

    output[non_empty_indices] = encoded
    return output


def build_weighted_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    """
    Weighted sum of per-field embeddings, then final L2 normalization.
    Missing fields contribute zero weight.
    """
    dim = model.get_sentence_embedding_dimension()
    batch_size_rows = len(df)

    weighted_sum = np.zeros((batch_size_rows, dim), dtype=np.float32)
    weight_sum = np.zeros((batch_size_rows, 1), dtype=np.float32)

    for field_name, field_weight in FIELD_WEIGHTS.items():
        if field_name not in df.columns:
            continue

        field_texts = [
            format_field_text(field_name, value)
            for value in df[field_name].tolist()
        ]

        field_emb = encode_texts(model, field_texts, batch_size=batch_size)

        # Row gets this weight only if the field is non-empty
        present_mask = np.array(
            [[1.0] if text.strip() else [0.0] for text in field_texts],
            dtype=np.float32,
        )

        weighted_sum += field_emb * field_weight
        weight_sum += present_mask * field_weight

    # Avoid division by zero; rows with no content should not exist, but be safe
    weight_sum = np.maximum(weight_sum, 1e-12)
    final_emb = weighted_sum / weight_sum
    final_emb = l2_normalize(final_emb).astype(np.float32)

    return final_emb


# ============================================================
# Main
# ============================================================
def main() -> None:
    resolved_input = resolve_input_path(INPUT_PARQUET)
    if not path_exists(INPUT_PARQUET):
        raise FileNotFoundError(f"Input file not found: {resolved_input}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )

    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model embedding dimension: {embedding_dim}")

    if resolved_input.startswith("s3://"):
        fs = s3_filesystem()
        parquet_file = pq.ParquetFile(fs.open(resolved_input.replace("s3://", ""), "rb"))
    else:
        parquet_file = pq.ParquetFile(resolved_input)

    temp_output = tempfile.NamedTemporaryFile(prefix="movie_embeddings_", suffix=".parquet", delete=False)
    temp_output_path = temp_output.name
    temp_output.close()

    writer = None
    total_rows = 0

    for batch_idx, record_batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE), start=1):
        df = record_batch.to_pandas()

        embeddings = build_weighted_embeddings(df, model, batch_size=BATCH_SIZE)

        output_dict: dict[str, Any] = {}

        for col in OUTPUT_METADATA_COLUMNS:
            if col in df.columns:
                output_dict[col] = df[col].tolist()

        # store vector as list<float>
        output_dict["embedding"] = [row.tolist() for row in embeddings]

        output_df = pd.DataFrame(output_dict)
        table = pa.Table.from_pandas(output_df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(temp_output_path, table.schema)

        writer.write_table(table)
        total_rows += len(output_df)

        print(f"Processed batch {batch_idx}, rows={len(output_df)}, total_rows={total_rows}")

    if writer is not None:
        writer.close()

    output_uri = upload_file_to_path(temp_output_path, OUTPUT_PARQUET)
    os.remove(temp_output_path)

    print(f"Done. Wrote embeddings to: {output_uri}")
    print(f"Output target mapping: {to_s3_uri(OUTPUT_PARQUET)}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()