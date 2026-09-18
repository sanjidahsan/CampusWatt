# CampusWatt — GridWise LLM API

BUP CSE Fest 2026 Hackathon (Online Preliminary) submission. An HTTP service that takes a 24-hour campus energy scenario plus 1–3 natural-language operator notes and returns the cheapest feasible 24-hour dispatch schedule.

```
Request → FastAPI → Pydantic Validation → Groq LLM (openai/gpt-oss-20b) →
Guardrail Validator → PuLP LP Optimizer (CBC) → Replay Validator → Response
```

The LLM is the sole interpreter of operator notes. Its output is treated as untrusted until deterministic guardrails accept it, and the final schedule is independently replayed before responding — a schedule that violates any rule is never returned.

## Requirements

- Python 3.11
- Pinned dependencies (`requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.111.0 | HTTP service |
| `uvicorn[standard]` | 0.29.0 | ASGI server |
| `pydantic` | 2.7.1 | Request/response validation |
| `groq` | 0.9.0 | LLM client |
| `PuLP` | 2.8.0 | LP modeling (bundled CBC solver, no separate install) |
| `python-dotenv` | 1.0.1 | `.env` loading |
| `httpx` | 0.27.2 | HTTP transport (pinned: `groq==0.9.0` cannot construct its client with httpx ≥ 0.28) |

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key. Never commit this value |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Model used for operator-note interpretation |
| `PORT` | No | `8000` | Service port |

## Run locally

```bash
git clone https://github.com/sanjidahsan/CampusWatt.git
cd CampusWatt
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Windows: copy .env.example .env
# Edit .env and set GROQ_API_KEY to your key
python main.py
```

Run the test suite (no API key needed):

```bash
python -m unittest discover -s tests -v
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Readiness check, returns `{"status": "ok"}` |
| POST | `/optimize-energy` | Interprets notes and returns the optimal 24-hour schedule |

Status codes: `200` on success; `400` on malformed JSON or any request-validation failure (semantic errors are mapped to `400`, not `422`); `500` as a controlled internal error (LLM, guardrail, optimizer, or replay failure) that never includes secrets or stack traces.

### Health check

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Optimize with the public sample

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Expected for `sample_request.json` (`SAMPLE-01`): note 0 → `solar_reduction` over hours `[12, 13]` with `factor 0.25`; note 1 → `no_op`; plus the cheapest valid 24-hour `hourly_plan` whose recalculated values match `total_grid_kwh`, `total_cost_bdt`, and `peak_grid_kwh`.

## How it works

### LLM role
The Groq model (`openai/gpt-oss-20b`, `temperature=0`) classifies each operator note into exactly one directive type — `solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`, `max_grid_window`, or `no_op` — with machine-checkable hours and numeric values. Time windows are start-inclusive/end-exclusive (`"1 PM to 3 PM"` → `[13, 14]`); solar factors are remaining fractions (`"80% reduction"` → `0.2`). See `interpreter.py`.

### Guardrails
Every field of the raw LLM output is validated deterministically before it reaches the optimizer: directive-type enum, note ordering, boolean `applies` semantics, unique ascending hours in 0–23, and per-type numeric ranges. Anything invalid fails safe with a controlled error — constraints are never invented or patched. See `guardrails.py`.

### Optimizer
A PuLP mixed-integer program minimizes total grid cost subject to energy balance, effective-solar caps, battery capacity/reserve/rate limits, binary charge/discharge mutual exclusion, all active directives, and end-of-day battery neutrality. Overlapping solar reductions combine multiplicatively (conservative). Solve capped at 25 seconds. See `optimizer.py`.

### Replay validator
The produced schedule is re-checked hour by hour against the same rules the judge applies (tolerance 0.01). Any violation fails the request instead of returning a bad plan. See `replay.py`.

## Docker

> TODO before submission: publish the image and replace the placeholders below with the exact registry reference (tag or digest).

```bash
docker pull <registry>/<image>:<tag>
docker run -e GROQ_API_KEY=your_key_here -p 8000:8000 <registry>/<image>:<tag>
curl http://localhost:8000/health
```

Build and run locally:

```bash
docker build -t gridwise:latest .
docker run -e GROQ_API_KEY=your_key_here -p 8000:8000 gridwise:latest
```

The image contains no baked-in secrets — `GROQ_API_KEY` is passed at runtime via `-e`, and `.dockerignore` keeps `.env` out of the build context. The service binds `0.0.0.0:8000`.

## Known limitations

1. The Groq API must be reachable during judging; the deployment needs a valid key with sufficient quota.
2. LP solve time is bounded to 25 seconds; scenarios with many tight constraints may approach this limit.
3. Groq `429` rate limits are retried twice (5s, 10s backoff); sustained quota exhaustion still returns a controlled `500`.
4. Numeric values use floating-point; differences within 0.01 kWh/BDT are treated as equivalent per spec.

## Credits

- [PuLP](https://github.com/coin-or/pulp) (CBC solver), [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [Groq SDK](https://github.com/groq/groq-python), [httpx](https://www.python-httpx.org/)
- Challenge specification, scoring rubric, and public sample cases: BUP CSE Fest 2026 organizers
- Developed with AI coding assistance (OpenCode)
