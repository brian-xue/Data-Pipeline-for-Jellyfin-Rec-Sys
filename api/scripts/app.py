from __future__ import annotations

import json
import os
from datetime import datetime

import psycopg2
from flask import Flask, jsonify, request


def create_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "recsys"),
        user=os.getenv("POSTGRES_USER", "recsys"),
        password=os.getenv("POSTGRES_PASSWORD", "recsys"),
        connect_timeout=5,
    )


def parse_event_time(value: str | None) -> datetime:
    if not value:
        raise ValueError("missing event_time")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def ensure_user_exists(cur, user_id: int) -> None:
    cur.execute(
        """
        INSERT INTO users (user_id, status, first_seen_at, last_seen_at)
        VALUES (%s, 'active', NOW(), NOW())
        ON CONFLICT (user_id) DO NOTHING
        """,
        (user_id,),
    )


def ensure_session_exists(cur, user_id: int, session_id: str, event_time: datetime) -> None:
    cur.execute(
        """
        INSERT INTO online_sessions (
            session_id,
            user_id,
            session_start_time,
            status,
            event_count,
            total_watch_duration_seconds
        )
        VALUES (%s, %s, %s, 'active', 0, 0.0)
        ON CONFLICT (session_id) DO NOTHING
        """,
        (session_id, user_id, event_time),
    )

    cur.execute("SELECT user_id FROM online_sessions WHERE session_id = %s", (session_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"session_id '{session_id}' does not exist after ensure")

    bound_user_id = int(row[0])
    if bound_user_id != int(user_id):
        raise ValueError(
            f"session_id '{session_id}' is already bound to user_id={bound_user_id}, "
            f"cannot use with user_id={user_id}"
        )


def insert_auth_event(cur, event: dict) -> None:
    user_id = int(event["user_id"])
    session_id = str(event["session_id"])
    event_type = str(event["event_type"])
    event_time = parse_event_time(event.get("event_time"))
    metadata_json = event.get("metadata_json") or {}

    if event_type not in {"login", "logout"}:
        raise ValueError(f"invalid auth event_type: {event_type}")

    ensure_user_exists(cur, user_id)

    if event_type == "login":
        ensure_session_exists(cur, user_id=user_id, session_id=session_id, event_time=event_time)
    else:
        cur.execute(
            """
            UPDATE online_sessions
            SET status = 'closed',
                session_end_time = %s,
                updated_at = NOW()
            WHERE session_id = %s
            """,
            (event_time, session_id),
        )

    cur.execute(
        """
        INSERT INTO auth_events (user_id, session_id, event_type, event_time, metadata_json)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        """,
        (user_id, session_id, event_type, event_time, json.dumps(metadata_json)),
    )


def insert_user_event(cur, event: dict) -> None:
    user_id = int(event["user_id"])
    movie_id = int(event["movie_id"])
    session_id = str(event["session_id"])
    event_time = parse_event_time(event.get("event_time"))
    watch_duration_seconds = float(event["watch_duration_seconds"])

    ensure_user_exists(cur, user_id)
    ensure_session_exists(cur, user_id=user_id, session_id=session_id, event_time=event_time)

    cur.execute(
        """
        INSERT INTO user_events (
            user_id,
            movie_id,
            session_id,
            event_type,
            event_time,
            watch_duration_seconds
        )
        VALUES (%s, %s, %s, 'finish', %s, %s)
        """,
        (user_id, movie_id, session_id, event_time, watch_duration_seconds),
    )

    cur.execute(
        """
        UPDATE online_sessions
        SET last_event_time = %s,
            event_count = event_count + 1,
            total_watch_duration_seconds = total_watch_duration_seconds + %s,
            updated_at = NOW()
        WHERE session_id = %s
        """,
        (event_time, watch_duration_seconds, session_id),
    )

    cur.execute(
        """
        UPDATE users
        SET last_seen_at = %s,
            updated_at = NOW()
        WHERE user_id = %s
        """,
        (event_time, user_id),
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/ingest/events")
    def ingest_events():
        payload = request.get_json(silent=True) or {}
        auth_events = payload.get("auth_events") or []
        user_events = payload.get("user_events") or []

        if not isinstance(auth_events, list) or not isinstance(user_events, list):
            return jsonify({"error": "auth_events and user_events must be arrays"}), 400

        try:
            with create_connection() as conn:
                with conn.cursor() as cur:
                    for item in auth_events:
                        if not isinstance(item, dict):
                            raise ValueError("each auth_events item must be object")
                        insert_auth_event(cur, item)

                    for item in user_events:
                        if not isinstance(item, dict):
                            raise ValueError("each user_events item must be object")
                        insert_user_event(cur, item)
                conn.commit()

            return jsonify(
                {
                    "status": "ok",
                    "auth_events_written": len(auth_events),
                    "user_events_written": len(user_events),
                }
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("event ingestion failed")
            return jsonify({"error": str(e)}), 500

    return app
