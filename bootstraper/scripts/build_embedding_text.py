from __future__ import annotations

import ast
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from minio_s3 import (
    path_exists,
    read_csv_auto,
    write_dataframe_jsonl_to_path,
    write_dataframe_parquet_to_path,
)


# =========================
# Config
# =========================
OBJECT_ROOT = Path("/data/")

RAW_BUCKET = "s3://raw"
CLEANED_BUCKET = "s3://cleaned"

MOVIES_FILE = f"{RAW_BUCKET}/movies.csv"
LINKS_FILE = f"{RAW_BUCKET}/links.csv"
TAGS_FILE = f"{RAW_BUCKET}/tags.csv"
TMDB_FILE = f"{RAW_BUCKET}/TMDB_movie_dataset_v11.csv"

OUTPUT_PARQUET = f"{CLEANED_BUCKET}/movie_embedding_text.parquet"
OUTPUT_JSONL = f"{CLEANED_BUCKET}/movie_embedding_text.jsonl"

TAGS_CHUNKSIZE = 250_000
TMDB_CHUNKSIZE = 100_000
TOP_K_TAGS = 15
MAX_OVERVIEW_CHARS = 2000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# Helpers
# =========================
def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def clean_text(text: Any) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        cleaned = clean_text(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result



def ensure_str_list(value: Any) -> list[str]:
    """
    Convert a value into a clean list[str].
    Handles None, NaN, scalar strings, lists, tuples, and set-like values.
    """
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []

    if isinstance(value, (list, tuple, set)):
        result = [clean_text(x) for x in value]
        return [x for x in result if x]

    text = clean_text(value)
    return [text] if text else []



def parse_movielens_genres(value: Any) -> list[str]:
    text = clean_text(value)
    if not text or text == "(no genres listed)":
        return []
    return dedupe_keep_order(text.split("|"))



def safe_literal_eval(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return None



def parse_jsonish_list(value: Any) -> list[str]:
    """
    Handles cases like:
    - '[{"id": 1, "name": "Animation"}]'
    - '["Animation", "Comedy"]'
    - "Animation|Comedy"
    - "Animation, Comedy"
    - already a list
    """
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return dedupe_keep_order(value)

    text = str(value).strip()
    if not text:
        return []

    parsed = None
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = safe_literal_eval(text)

    if parsed is not None:
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            result: list[str] = []
            for item in parsed:
                if isinstance(item, dict):
                    name = (
                        item.get("name")
                        or item.get("english_name")
                        or item.get("iso_639_1")
                    )
                    if name:
                        result.append(str(name))
                else:
                    result.append(str(item))
            return dedupe_keep_order(result)

    if "|" in text:
        return dedupe_keep_order(text.split("|"))
    if "," in text:
        return dedupe_keep_order(text.split(","))

    text = clean_text(text)
    return [text] if text else []



def to_yes_no(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return "Yes" if value != 0 else "No"
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "1", "yes"}:
            return "Yes"
        if val in {"false", "0", "no"}:
            return "No"
    return ""



def extract_release_year(release_date: Any, title: Any) -> str:
    rd = clean_text(release_date)
    if rd:
        match = re.match(r"^(\d{4})", rd)
        if match:
            return match.group(1)

    title_text = clean_text(title)
    match = re.search(r"\((\d{4})\)\s*$", title_text)
    if match:
        return match.group(1)

    return ""



def has_enough_content(row: pd.Series) -> bool:
    title = clean_text(row.get("title"))
    genres = ensure_str_list(row.get("genres_list"))
    overview = clean_text(row.get("overview"))
    tagline = clean_text(row.get("tagline"))
    keywords = ensure_str_list(row.get("keywords_list"))
    top_tags = ensure_str_list(row.get("top_user_tags"))

    if not title:
        return False

    semantic_signals = 0
    if genres:
        semantic_signals += 1
    if overview:
        semantic_signals += 1
    if tagline:
        semantic_signals += 1
    if keywords:
        semantic_signals += 1
    if top_tags:
        semantic_signals += 1

    return semantic_signals >= 1



def build_embedding_text(row: pd.Series) -> str:
    lines: list[str] = []

    title = clean_text(row.get("title"))
    if title:
        lines.append(f"Title: {title}")

    original_title = clean_text(row.get("original_title"))
    if original_title and original_title.lower() != title.lower():
        lines.append(f"Original title: {original_title}")

    genres = ensure_str_list(row.get("genres_list"))
    if genres:
        lines.append(f"Genres: {', '.join(genres)}")

    tagline = clean_text(row.get("tagline"))
    if tagline:
        lines.append(f"Tagline: {tagline}")

    overview = clean_text(row.get("overview"))
    if overview:
        lines.append(f"Overview: {overview[:MAX_OVERVIEW_CHARS].strip()}")

    keywords = ensure_str_list(row.get("keywords_list"))
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")

    top_user_tags = ensure_str_list(row.get("top_user_tags"))
    if top_user_tags:
        lines.append(f"User tags: {', '.join(top_user_tags)}")

    original_language = clean_text(row.get("original_language"))
    if original_language:
        lines.append(f"Original language: {original_language}")

    spoken_languages = ensure_str_list(row.get("spoken_languages_list"))
    if spoken_languages:
        lines.append(f"Spoken languages: {', '.join(spoken_languages)}")

    production_countries = ensure_str_list(row.get("production_countries_list"))
    if production_countries:
        lines.append(f"Production countries: {', '.join(production_countries)}")

    production_companies = ensure_str_list(row.get("production_companies_list"))
    if production_companies:
        lines.append(f"Production companies: {', '.join(production_companies)}")

    adult = to_yes_no(row.get("adult"))
    if adult:
        lines.append(f"Adult: {adult}")

    release_year = clean_text(row.get("release_year"))
    if release_year:
        lines.append(f"Release year: {release_year}")

    return "\n".join(lines).strip()


# =========================
# Loaders
# =========================
def ensure_required_files() -> None:
    required = [MOVIES_FILE, LINKS_FILE, TAGS_FILE, TMDB_FILE]
    missing = [str(path) for path in required if not path_exists(path)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))



def load_movies() -> pd.DataFrame:
    logger.info("Loading movies.csv ...")
    df = read_csv_auto(
        MOVIES_FILE,
        usecols=["movieId", "title", "genres"],
        dtype={
            "movieId": "Int64",
            "title": "string",
            "genres": "string",
        },
    )
    df["title"] = df["title"].map(clean_text)
    df["genres_list"] = df["genres"].map(parse_movielens_genres)
    return df[["movieId", "title", "genres_list"]]



def load_links() -> pd.DataFrame:
    logger.info("Loading links.csv ...")
    df = read_csv_auto(
        LINKS_FILE,
        usecols=["movieId", "imdbId", "tmdbId"],
        dtype={
            "movieId": "Int64",
            "imdbId": "string",
            "tmdbId": "Int64",
        },
    )
    return df



def aggregate_tags_chunked() -> pd.DataFrame:
    logger.info("Aggregating tags.csv in chunks ...")
    tag_counter_by_movie: dict[int, Counter[str]] = defaultdict(Counter)

    for i, chunk in enumerate(
        read_csv_auto(
            TAGS_FILE,
            usecols=["movieId", "tag"],
            dtype={
                "movieId": "Int64",
                "tag": "string",
            },
            chunksize=TAGS_CHUNKSIZE,
        ),
        start=1,
    ):
        logger.info("Processing tags chunk %d ...", i)
        chunk["tag"] = chunk["tag"].map(clean_text).str.lower()
        chunk = chunk[(chunk["movieId"].notna()) & (chunk["tag"] != "")]
        if chunk.empty:
            continue

        grouped = chunk.groupby(["movieId", "tag"]).size()
        for (movie_id, tag), count in grouped.items():
            if pd.isna(movie_id):
                continue
            tag_counter_by_movie[int(movie_id)][tag] += int(count)

    rows = []
    for movie_id, counter in tag_counter_by_movie.items():
        top_tags = [tag for tag, _ in counter.most_common(TOP_K_TAGS)]
        rows.append({"movieId": movie_id, "top_user_tags": top_tags})

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=["movieId", "top_user_tags"])
    else:
        result["movieId"] = result["movieId"].astype("Int64")

    logger.info("Aggregated tags for %d movies.", len(result))
    return result



def load_tmdb_filtered(needed_tmdb_ids: set[int]) -> pd.DataFrame:
    logger.info("Loading TMDB file in chunks and filtering by tmdbId ...")

    tmdb_usecols = [
        "id",
        "title",
        "original_title",
        "overview",
        "tagline",
        "genres",
        "keywords",
        "original_language",
        "spoken_languages",
        "production_countries",
        "production_companies",
        "adult",
        "release_date",
    ]

    collected_chunks: list[pd.DataFrame] = []

    for i, chunk in enumerate(
        read_csv_auto(
            TMDB_FILE,
            usecols=lambda c: c in tmdb_usecols,
            dtype={
                "id": "Int64",
                "title": "string",
                "original_title": "string",
                "overview": "string",
                "tagline": "string",
                "genres": "string",
                "keywords": "string",
                "original_language": "string",
                "spoken_languages": "string",
                "production_countries": "string",
                "production_companies": "string",
                "adult": "string",
                "release_date": "string",
            },
            chunksize=TMDB_CHUNKSIZE,
        ),
        start=1,
    ):
        logger.info("Processing TMDB chunk %d ...", i)
        chunk = chunk[chunk["id"].isin(needed_tmdb_ids)]
        if chunk.empty:
            continue
        collected_chunks.append(chunk)

    if not collected_chunks:
        logger.warning("No TMDB rows matched the needed tmdbId values.")
        return pd.DataFrame(
            columns=[
                "id",
                "title_tmdb",
                "original_title",
                "overview",
                "tagline",
                "tmdb_genres_list",
                "keywords_list",
                "original_language",
                "spoken_languages_list",
                "production_countries_list",
                "production_companies_list",
                "adult",
                "release_date",
            ]
        )

    tmdb_df = pd.concat(collected_chunks, ignore_index=True)
    logger.info("Loaded %d matched TMDB rows.", len(tmdb_df))

    if "title" in tmdb_df.columns:
        tmdb_df = tmdb_df.rename(columns={"title": "title_tmdb"})

    for col in ["title_tmdb", "original_title", "overview", "tagline", "original_language", "release_date"]:
        if col in tmdb_df.columns:
            tmdb_df[col] = tmdb_df[col].map(clean_text)

    tmdb_df["tmdb_genres_list"] = tmdb_df["genres"].map(parse_jsonish_list) if "genres" in tmdb_df.columns else [[] for _ in range(len(tmdb_df))]
    tmdb_df["keywords_list"] = tmdb_df["keywords"].map(parse_jsonish_list) if "keywords" in tmdb_df.columns else [[] for _ in range(len(tmdb_df))]
    tmdb_df["spoken_languages_list"] = tmdb_df["spoken_languages"].map(parse_jsonish_list) if "spoken_languages" in tmdb_df.columns else [[] for _ in range(len(tmdb_df))]
    tmdb_df["production_countries_list"] = tmdb_df["production_countries"].map(parse_jsonish_list) if "production_countries" in tmdb_df.columns else [[] for _ in range(len(tmdb_df))]
    tmdb_df["production_companies_list"] = tmdb_df["production_companies"].map(parse_jsonish_list) if "production_companies" in tmdb_df.columns else [[] for _ in range(len(tmdb_df))]

    keep_cols = [
        "id",
        "title_tmdb",
        "original_title",
        "overview",
        "tagline",
        "tmdb_genres_list",
        "keywords_list",
        "original_language",
        "spoken_languages_list",
        "production_countries_list",
        "production_companies_list",
        "adult",
        "release_date",
    ]
    return tmdb_df[keep_cols]


# =========================
# Main
# =========================
def main() -> None:
    ensure_required_files()

    movies_df = load_movies()
    links_df = load_links()
    tags_df = aggregate_tags_chunked()

    needed_tmdb_ids = {
        int(x)
        for x in links_df["tmdbId"].dropna().tolist()
        if pd.notna(x)
    }
    logger.info("Need to look up %d tmdbIds.", len(needed_tmdb_ids))

    tmdb_df = load_tmdb_filtered(needed_tmdb_ids)

    logger.info("Joining datasets ...")
    merged = movies_df.merge(
        links_df,
        on="movieId",
        how="left",
    )

    merged = merged.merge(
        tmdb_df,
        left_on="tmdbId",
        right_on="id",
        how="left",
    )

    merged = merged.merge(
        tags_df,
        on="movieId",
        how="left",
    )

    logger.info("Normalizing joined fields ...")

    list_columns = [
        "genres_list",
        "keywords_list",
        "top_user_tags",
        "spoken_languages_list",
        "production_countries_list",
        "production_companies_list",
    ]
    for col in list_columns:
        if col in merged.columns:
            merged[col] = merged[col].apply(ensure_str_list)

    merged["top_user_tags"] = merged["top_user_tags"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    merged["genres_list"] = merged.apply(
        lambda row: row["tmdb_genres_list"]
        if isinstance(row.get("tmdb_genres_list"), list) and len(row["tmdb_genres_list"]) > 0
        else ensure_str_list(row.get("genres_list")),
        axis=1,
    )

    merged["title"] = merged.apply(
        lambda row: clean_text(row.get("title")) or clean_text(row.get("title_tmdb")),
        axis=1,
    )

    merged["release_year"] = merged.apply(
        lambda row: extract_release_year(row.get("release_date"), row.get("title")),
        axis=1,
    )

    logger.info("Building embedding text ...")
    merged["embedding_text"] = merged.apply(build_embedding_text, axis=1)

    logger.info("Applying minimum-content filter ...")
    before = len(merged)
    merged = merged[merged.apply(has_enough_content, axis=1)].copy()
    merged = merged[merged["embedding_text"].str.len() > 0].copy()
    after = len(merged)

    logger.info("Kept %d / %d movies after filtering.", after, before)

    output_cols = [
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
        "spoken_languages_list",
        "production_countries_list",
        "production_companies_list",
        "tagline",
        "overview",
        "embedding_text",
    ]
    output_cols = [c for c in output_cols if c in merged.columns]
    result = merged[output_cols].copy()

    logger.info("Writing Parquet to MinIO target mapped from %s ...", OUTPUT_PARQUET)
    parquet_uri = write_dataframe_parquet_to_path(result, OUTPUT_PARQUET, index=False)

    logger.info("Writing JSONL to MinIO target mapped from %s ...", OUTPUT_JSONL)
    jsonl_uri = write_dataframe_jsonl_to_path(result, OUTPUT_JSONL)

    logger.info(
        "Done. Generated %d cleaned movie rows. Uploaded to %s and %s",
        len(result),
        parquet_uri,
        jsonl_uri,
    )


if __name__ == "__main__":
    main()
