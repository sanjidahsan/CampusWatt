# GridWise LLM API

## Problem
Campus microgrid must serve a 24-hour demand profile with rooftop solar, grid import at hourly tariffs, and a battery. Operator notes in natural language (e.g. panel cleaning, reserve requirements, charge windows) constrain the schedule. This service interprets notes via Groq LLM, validates them deterministically, and produces the cheapest feasible 24-hour dispatch with PuLP/CBC.

## Architecture
```
Request → FastAPI → Pydantic Validation → Groq LLM (llama3-70b-8192) →
Guardrail Validator → PuLP LP Optimizer (CBC) → Replay Validator → Response
```

## Setup (local)
```bash
git clone <repo-url>
cd gridwise
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key
python main.py
```

## Environment Variables
| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key |
| `GROQ_MODEL` | No | `llama3-70b-8192` | Groq model to use |
| `PORT` | No | `8000` | Service port |

## Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

## Sample Request
```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Docker
```bash
docker pull <registry>/<image>:<tag>
docker run -e GROQ_API_KEY=your_key_here -p 8000:8000 <registry>/<image>:<tag>
curl http://localhost:8000/health
```

Build locally:
```bash
docker build -t gridwise:latest .
docker run -e GROQ_API_KEY=your_key_here -p 8000:8000 gridwise:latest
```

## LLM Role
Groq LLM (`llama3-70b-8192`) interprets each operator note into a structured directive at `temperature=0`. Output is validated deterministically by guardrails before being applied to the LP optimizer.

## Optimizer
PuLP with the bundled CBC MILP solver. Binary variables enforce mutual exclusion of charge/discharge.

## Guardrails
All LLM output is treated as untrusted. Deterministic validation checks types, ranges, hour uniqueness, and directive-specific numeric constraints before any LLM output touches the optimizer.

## Known Limitations
- Groq API must be reachable during judging; ensure valid API key and quota
- LP solve time is bounded to 25 seconds; scenarios with many tight constraints may approach this limit
- Numeric values use floating-point; differences within 0.01 kWh/BDT are treated as equivalent per spec
# CampusWatt
