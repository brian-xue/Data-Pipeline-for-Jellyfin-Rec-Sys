from __future__ import annotations

from typing import Any

from psycopg2.extras import RealDictCursor


class PopularMovieRepository:
	def __init__(self, conn):
		self.conn = conn

	def replace_popular_movies(self, movies: list[dict[str, Any]]) -> None:
		with self.conn.cursor() as cur:
			cur.execute("DELETE FROM popular_movies")
			for item in movies:
				cur.execute(
					"""
					INSERT INTO popular_movies (rank_position, movie_id, score, updated_at)
					VALUES (%s, %s, %s, NOW())
					""",
					(item["rank_position"], item["movie_id"], item["score"]),
				)

	def fetch_popular_movies(self, limit: int = 50) -> list[dict[str, Any]]:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"""
				SELECT rank_position, movie_id, score, updated_at
				FROM popular_movies
				ORDER BY rank_position ASC
				LIMIT %s
				""",
				(limit,),
			)
			return [dict(row) for row in cur.fetchall()]
