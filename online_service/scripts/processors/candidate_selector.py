from __future__ import annotations

from scripts.db.connection import get_connection
from scripts.repositories.popular_movie_repository import PopularMovieRepository
from scripts.repositories.user_repository import UserRepository
from scripts.services.candidate_service import retrieve_by_embedding, retrieve_from_popular


POPULAR_FALLBACK_COUNT = 20

def select_candidates_for_user(user_id: int, top_k: int, config) -> Tuple[str, list[dict]]:
    with get_connection(config) as conn:
        user_repo = UserRepository(conn)
        popular_repo = PopularMovieRepository(conn)

        user_row = user_repo.get_user_by_id(user_id)
        if user_row is None or not user_row.get("embedding_uri"):
            print(f"[candidate selection] using popular movies for user_id={user_id}")
            popular_rows = popular_repo.fetch_popular_movies(limit=POPULAR_FALLBACK_COUNT)
            return ("popular", retrieve_from_popular(popular_rows, top_k=min(POPULAR_FALLBACK_COUNT, len(popular_rows))), None)

        print(f"[candidate selection] using embedding for user_id={user_id}")
        embedding_items, user_embedding = retrieve_by_embedding(
            user_id=user_id,
            user_embedding_uri=user_row.get("embedding_uri"),
            top_k=top_k,
            config=config,
        )
        if embedding_items:
            return ("embedding", embedding_items, user_embedding)

        popular_rows = popular_repo.fetch_popular_movies(limit=POPULAR_FALLBACK_COUNT)
        return ("popular", retrieve_from_popular(popular_rows, top_k=min(POPULAR_FALLBACK_COUNT, len(popular_rows))), None)