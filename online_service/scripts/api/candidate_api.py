from __future__ import annotations

from flask import Flask, jsonify, request

from scripts.config.config import load_online_service_config
from scripts.processors.candidate_selector import select_candidates_for_user


def create_app(config=None) -> Flask:
    cfg = config or load_online_service_config()
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/candidates")
    def candidates():
        user_id_raw = request.args.get("user_id")
        if user_id_raw is None:
            return jsonify({"error": "missing user_id"}), 400

        top_k_raw = request.args.get("top_k")
        top_k = int(top_k_raw) if top_k_raw else cfg.candidate.top_k_default

        try:
            user_id = int(user_id_raw)
            category, items, user_embedding = select_candidates_for_user(user_id=user_id, top_k=top_k, config=cfg)
            return jsonify({
                "user_id": user_id,
                "category": category,
                "top_k": top_k,
                "count": len(items),
                "items": items,
                "user_embedding": user_embedding,
            })
        except Exception as e:
            app.logger.exception("candidate selection failed for user_id=%s top_k=%s", user_id, top_k)
            return jsonify({"error": str(e)}), 500

    return app
