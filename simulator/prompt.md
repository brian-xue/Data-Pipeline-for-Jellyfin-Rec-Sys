You are asked to implement a configurable **event simulator system** for a recommendation system backend.

The simulator should generate synthetic user behavior data and write into two PostgreSQL tables:

* `auth_events` (login/logout)
* `user_events` (movie watching events, event_type = 'finish')

The system must be **fully configurable via a YAML file**, and the codebase must be modular, maintainable, and extensible.
要求直接写入postgresql
---

# 🧱 Overall Requirements

1. The simulator must:

   * Maintain a set of **online users**
   * Ensure only logged-in users can generate `user_events`
   * Simulate login, logout, and watching behavior
   * Generate realistic event timestamps and session IDs

2. The simulator must support:

   * Configurable number of online users
   * Configurable event generation frequency
   * Configurable number of events per user session
   * Configurable user pool size
   * Configurable movie ID range

3. All configuration MUST come from a YAML file (no hardcoded parameters)

---

# 📁 Required Project Structure

```
simulator/scripts/
├── config.py
├── main.py
├── runner.py
├── state.py
├── scheduler.py
├── db/
│   ├── client.py
│   └── writer.py
├── generators/
│   ├── user_pool.py
│   ├── auth_generator.py
│   └── event_generator.py
└── services/
    └── simulator_service.py
```

---

# 📌 Module Responsibilities

## config.py

* Load configuration from a YAML file
* Convert YAML into Python dataclasses
* Must include:

  * database config
  * user pool config
  * online user config
  * event rate config
  * per-user behavior config

---

## state.py

Maintain runtime state (in-memory only):

* online_users: Set[user_id]
* offline_users: Set[user_id]
* runtime_by_user:

  * session_id
  * events_in_session

---

## scheduler.py

Controls simulation timing:

* Runs in ticks (e.g., every 1 second)
* Each tick should:

  1. Ensure target number of online users
  2. Generate user events
  3. Logout some users

---

## db/client.py

* Create PostgreSQL connection using SQLAlchemy

## db/writer.py

* Provide ONLY these methods:

  * insert_auth_event(...)
  * insert_user_event(...)
  * commit()

No business logic should exist here.

---

## generators/

### user_pool.py

* Generate a fixed pool of user_ids based on config

### auth_generator.py

* Generate login/logout event payloads

### event_generator.py

* Generate user_events:

  * movie_id must be within configured range
  * event_type must always be "finish"

---

## services/simulator_service.py

This is the CORE module.

It must implement:

### login_user(user_id)

* Insert auth_event (login)
* Add user to online_users
* Create session_id

### logout_user(user_id)

* Insert auth_event (logout)
* Remove user from online_users

### emit_one_user_event(user_id)

* ONLY allowed if user is online
* Insert user_event
* Increase event counter

### ensure_target_online_users()

* Bring online users up to target_online_users

### emit_user_events_for_tick()

* Generate events based on:

  * global_event_rate_per_tick
  * per_user_event_prob
  * min/max events per session

### logout_some_users()

* Logout users based on config

### run_tick()

* Execute full simulation step

---

## runner.py

* Initialize:

  * user pool
  * state
  * service
* Run simulation loop:

  * for each tick:

    * run_tick()
    * commit()
    * sleep()

---

## main.py

* Entry point
* Load YAML config
* Start simulator

---

# ⚙️ YAML Configuration Requirements

The YAML config MUST support:

```yaml
postgres:
  host: localhost
  port: 5432
  dbname: recsys
  user: recsys
  password: recsys123

simulator:
  user_pool_size: 5000
  min_user_id: 10000000
  max_user_id: 99999999

  target_online_users: 100
  max_online_users: 120

  tick_seconds: 1.0
  total_ticks: 3600

  login_rate_per_tick: 10
  logout_rate_per_tick: 5

  global_event_rate_per_tick: 50
  per_user_event_prob: 0.3

  min_events_per_session: 1
  max_events_per_session: 10

  min_movie_id: 1
  max_movie_id: 292757

  min_watch_duration_seconds: 60
  max_watch_duration_seconds: 7200
```

---

# ⚠️ Important Constraints

1. ONLY logged-in users can generate user_events
2. No database triggers — all logic must be handled in Python
3. No external dependencies beyond:

   * sqlalchemy
   * pyyaml
   * standard library
4. Code must be clean, modular, and production-style
使用postgres
---

# 🎯 Expected Output

Generate full Python implementation for all modules above, including:

* YAML loader
* Data classes
* Database writer
* Simulator logic
* Main entrypoint

The code should be runnable like:

```
python simulator/main.py --config config.yaml
```

---

Focus on clarity, correctness, and maintainability.
