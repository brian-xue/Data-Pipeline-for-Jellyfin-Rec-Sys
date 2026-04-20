CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',

    embedding_uri TEXT,
    embedding_version TEXT,
    embedding_updated_at TIMESTAMPTZ,

    login_count INTEGER NOT NULL DEFAULT 0,
    last_login_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    first_seen_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS online_sessions (
    session_id TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    session_start_time TIMESTAMPTZ NOT NULL,
    session_end_time TIMESTAMPTZ,
    last_event_time TIMESTAMPTZ,

    status TEXT NOT NULL DEFAULT 'active',
    event_count INTEGER NOT NULL DEFAULT 0,
    total_watch_duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS user_events (
    event_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    movie_id BIGINT NOT NULL,
    session_id TEXT REFERENCES online_sessions(session_id) ON DELETE SET NULL,

    event_type TEXT NOT NULL DEFAULT 'finish',
    event_time TIMESTAMPTZ NOT NULL,
    watch_duration_seconds DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS auth_events (
    auth_event_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id TEXT REFERENCES online_sessions(session_id) ON DELETE SET NULL,

    event_type TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS popular_movies (
    rank_position INTEGER PRIMARY KEY,
    movie_id BIGINT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_embedding_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    embedding_uri TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    model_version TEXT,
    source_event_max_id BIGINT,
    source_event_count INTEGER,
    embedding_updated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- CREATE TABLE IF NOT EXISTS session_events (
--     session_event_id BIGSERIAL PRIMARY KEY,
--     user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
--     session_id TEXT REFERENCES online_sessions(session_id) ON DELETE SET NULL,

--     event_type TEXT NOT NULL,
--     event_time TIMESTAMPTZ NOT NULL,

--     created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
--     metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
-- );

CREATE TABLE IF NOT EXISTS service_checkpoints (
    job_name TEXT PRIMARY KEY,

    last_auth_event_id BIGINT NOT NULL DEFAULT 0,
    last_user_event_id BIGINT NOT NULL DEFAULT 0,

    last_auth_event_time TIMESTAMPTZ,
    last_user_event_time TIMESTAMPTZ,

    status TEXT NOT NULL DEFAULT 'idle',

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS online_event_stats (
    window_start_time TIMESTAMPTZ PRIMARY KEY,
    window_end_time TIMESTAMPTZ NOT NULL,
    total_count BIGINT NOT NULL DEFAULT 0,
    abnormal_count BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


INSERT INTO service_checkpoints (job_name)
VALUES
    ('online_service_auth'),
    ('online_service_event'),
    ('online_service_popular'),
    ('online_service_export'),
    ('online_service_user_embedding')
ON CONFLICT (job_name) DO NOTHING;

-- CREATE INDEX IF NOT EXISTS idx_users_last_seen_at
--     ON users(last_seen_at);

-- CREATE INDEX IF NOT EXISTS idx_online_sessions_user_id
--     ON online_sessions(user_id);

-- CREATE INDEX IF NOT EXISTS idx_online_sessions_status
--     ON online_sessions(status);

-- CREATE INDEX IF NOT EXISTS idx_user_events_user_id
--     ON user_events(user_id);

-- CREATE INDEX IF NOT EXISTS idx_user_events_movie_id
--     ON user_events(movie_id);

-- CREATE INDEX IF NOT EXISTS idx_user_events_session_id
--     ON user_events(session_id);

-- CREATE INDEX IF NOT EXISTS idx_user_events_event_time
--     ON user_events(event_time);

-- CREATE INDEX IF NOT EXISTS idx_auth_events_user_id
--     ON auth_events(user_id);

-- CREATE INDEX IF NOT EXISTS idx_auth_events_session_id
--     ON auth_events(session_id);

-- CREATE INDEX IF NOT EXISTS idx_auth_events_event_time
--     ON auth_events(event_time);


ALTER TABLE user_events
    ADD CONSTRAINT chk_user_events_event_type
    CHECK (event_type = 'finish');

ALTER TABLE auth_events
    ADD CONSTRAINT chk_auth_events_event_type
    CHECK (event_type IN ('login', 'logout'));

ALTER TABLE online_sessions
    ADD CONSTRAINT chk_online_sessions_status
    CHECK (status IN ('active', 'closed'));

ALTER TABLE popular_movies
    ADD CONSTRAINT chk_popular_movies_rank_position
    CHECK (rank_position BETWEEN 1 AND 50);

ALTER TABLE user_events
    ADD CONSTRAINT chk_user_events_watch_duration
    CHECK (watch_duration_seconds >= 0);

ALTER TABLE users
    ADD CONSTRAINT chk_users_login_count
    CHECK (login_count >= 0);

ALTER TABLE online_event_stats
    ADD CONSTRAINT chk_online_event_stats_non_negative
    CHECK (total_count >= 0 AND abnormal_count >= 0 AND abnormal_count <= total_count);

-- ALTER TABLE session_events
--     ADD CONSTRAINT chk_session_events_event_type
--     CHECK (event_type IN ('refresh'));