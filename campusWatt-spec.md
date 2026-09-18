# GridWise LLM — Complete Project Specification
**BUP CSE Fest 2026 Hackathon · Online Preliminary**
**Version:** 1.0 — Single Source of Truth for Coding Agent

---

## 0. Overview

Build a deployed HTTP API that:
1. Receives a 24-hour campus energy scenario + 1–3 natural-language operator notes
2. Uses a **Groq LLM** to interpret those notes into structured directives
3. Validates the directives with deterministic guardrails
4. Runs a **linear programming optimizer** (PuLP + CBC) to produce the cheapest valid 24-hour energy schedule
5. Returns a fully compliant JSON response

**The pipeline is strictly sequential and non-negotiable:**
```
Request → Pydantic Validation → LLM Interpreter → Guardrail Validator → LP Optimizer → Replay Validator → Response
```

---

## 1. Project Structure

```
gridwise/
├── main.py               # FastAPI app, endpoint handlers
├── models.py             # All Pydantic request/response schemas
├── interpreter.py        # Groq LLM call + raw JSON extraction
├── guardrails.py         # Deterministic validation of LLM output
├── optimizer.py          # PuLP LP model, directive application, schedule builder
├── replay.py             # Post-optimization schedule verification
├── requirements.txt      # All dependencies pinned
├── Dockerfile            # Container definition
├── .env.example          # Env var names only, no values
├── .gitignore            # Must include .env
└── README.md             # Self-contained local quickstart
```

---

## 2. Dependencies (`requirements.txt`)

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
groq==0.9.0
PuLP==2.8.0
python-dotenv==1.0.1
```

> CBC solver ships bundled with PuLP — no separate installation required.

---

## 3. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key |
| `GROQ_MODEL` | No | Defaults to `"llama3-70b-8192"` |
| `PORT` | No | Defaults to `8000` |

Never commit values. Document names only in README and `.env.example`.

`.env.example`:
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
PORT=8000
```

---

## 4. API Contract

### 4.1 Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Readiness check |
| `/optimize-energy` | POST | Main LLM + optimization endpoint |

### 4.2 HTTP Status Codes

| Code | When |
|---|---|
| `200` | Successful health or optimization response |
| `400` | Malformed JSON or structurally invalid request |
| `422` | Well-formed but semantically invalid request (Pydantic default) |
| `500` | Controlled internal error — never expose secrets, stack traces, or raw LLM output |

### 4.3 GET /health Response
```json
{"status": "ok"}
```
Must return within 60 seconds of service start.

---

## 5. Pydantic Schemas (`models.py`)

### 5.1 Request Schema

```python
class HourEntry(BaseModel):
    hour: int                    # 0–23, unique across the 24 entries
    demand_kwh: float            # >= 0
    solar_kwh: float             # >= 0
    tariff_bdt_per_kwh: float    # >= 0

class BatteryConfig(BaseModel):
    capacity_kwh: float               # > 0
    initial_energy_kwh: float         # >= 0, <= capacity_kwh
    minimum_energy_kwh: float         # >= 0, < capacity_kwh
    max_charge_kwh_per_hour: float    # > 0
    max_discharge_kwh_per_hour: float # > 0

class OptimizeRequest(BaseModel):
    scenario_id: str                         # non-empty
    operator_notes: List[str]                # 1–3 items, each non-empty
    hours: List[HourEntry]                   # exactly 24 entries, hours 0–23
    battery: BatteryConfig

    @validator('operator_notes')
    def validate_notes(cls, v):
        assert 1 <= len(v) <= 3, "operator_notes must have 1–3 items"
        assert all(s.strip() for s in v), "operator_notes items must be non-empty"
        return v

    @validator('hours')
    def validate_hours(cls, v):
        assert len(v) == 24, "hours must contain exactly 24 entries"
        hour_vals = [h.hour for h in v]
        assert sorted(hour_vals) == list(range(24)), "hours must cover 0–23 uniquely"
        return sorted(v, key=lambda h: h.hour)  # sort by hour ascending
```

### 5.2 Response Schema

```python
class StructuredAdjustment(BaseModel):
    hours: List[int]
    factor: Optional[float] = None              # solar_reduction
    minimum_energy_kwh: Optional[float] = None  # minimum_battery_reserve
    max_grid_kwh: Optional[float] = None        # max_grid_window

class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: str   # one of the 6 allowed types
    structured_adjustment: Optional[StructuredAdjustment]  # null for no_op
    explanation: str

class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str         # "charge" | "discharge" | "idle"
    battery_kwh: float          # magnitude; 0 when idle
    battery_energy_after_kwh: float

class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
```

---

## 6. LLM Interpreter (`interpreter.py`)

### 6.1 Purpose
Call Groq API with a carefully structured prompt. Return raw parsed JSON list. Do NOT validate here — that is guardrails' job.

### 6.2 Groq Client Setup
```python
from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
```

### 6.3 System Prompt (embed verbatim)

```
You are an energy management system operator-note interpreter.

Your job is to read campus operator notes and classify each one into exactly one of these directive types:

1. solar_reduction      — reduce usable solar by a factor during specific hours
2. minimum_battery_reserve — keep battery energy at or above a level during specific hours
3. no_charge_window     — battery charging is unavailable during specific hours
4. no_discharge_window  — battery discharging is unavailable during specific hours
5. max_grid_window      — grid import must not exceed a value during specific hours
6. no_op                — the note does NOT affect the 24-hour energy schedule

CRITICAL RULES YOU MUST FOLLOW:

TIME WINDOWS:
- Time is expressed as whole-hour intervals
- The start hour is INCLUDED, the end hour is EXCLUDED
- "1 PM to 3 PM" → hours [13, 14]
- "2 AM until 5 AM" → hours [2, 3, 4]
- "noon until 2 PM" → hours [12, 13]
- "6 PM until 9 PM" → hours [18, 19, 20]
- "6 PM until 10 PM" → hours [18, 19, 20, 21]
- "7 PM until 9 PM" → hours [19, 20]
- Hours must always be unique integers 0–23, returned in ascending order

SOLAR REDUCTION FACTOR:
- factor = the REMAINING usable fraction (not the reduction amount)
- "80% reduction" → factor = 0.2
- "drop to about 20%" → factor = 0.2
- "roughly 25% of the forecast" → factor = 0.25
- "half of the forecast solar" → factor = 0.5
- "one-fifth of normal" → factor = 0.2
- factor must be between 0.0 and 1.0 inclusive

MINIMUM BATTERY RESERVE:
- Extract the kWh value directly when stated: "at least 120 kWh" → minimum_energy_kwh = 120
- Convert percentage to kWh when needed: "50% of battery capacity" with capacity=200 → minimum_energy_kwh = 100
- The battery capacity_kwh is provided in the context below

NO-OP CLASSIFICATION:
- If the note discusses anything unrelated to today's 24-hour energy schedule (cafeteria menus, sports deadlines, library hours, booking changes, administrative events), classify as no_op
- no_op must have applies = false and structured_adjustment = null

OUTPUT FORMAT:
Return ONLY a valid JSON array, no other text, no markdown, no explanation outside the JSON.
One object per note, in note_index order starting from 0.

Each object must have exactly these fields:
{
  "note_index": <integer>,
  "applies": <boolean>,
  "directive_type": <string from the 6 types above>,
  "structured_adjustment": <object or null>,
  "explanation": <string>
}

structured_adjustment shapes by directive type:
- solar_reduction:          {"hours": [...], "factor": <number 0-1>}
- minimum_battery_reserve:  {"hours": [...], "minimum_energy_kwh": <number>}
- no_charge_window:         {"hours": [...]}
- no_discharge_window:      {"hours": [...]}
- max_grid_window:          {"hours": [...], "max_grid_kwh": <number>}
- no_op:                    null

NEVER invent directive types not listed above.
NEVER output anything outside the JSON array.
```

### 6.4 User Prompt Template

```python
def build_user_prompt(notes: List[str], battery: BatteryConfig) -> str:
    notes_text = "\n".join(
        f"{i}. \"{note}\"" for i, note in enumerate(notes)
    )
    return f"""Battery capacity for this scenario: {battery.capacity_kwh} kWh
Initial battery energy: {battery.initial_energy_kwh} kWh

Operator notes to interpret ({len(notes)} total):
{notes_text}

Return a JSON array with exactly {len(notes)} objects, one per note, in note_index order 0 to {len(notes)-1}."""
```

### 6.5 API Call

```python
def call_llm(notes: List[str], battery: BatteryConfig) -> List[dict]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(notes, battery)}
        ],
        temperature=0.0,    # deterministic
        max_tokens=2048,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)   # raises json.JSONDecodeError on failure
```

### 6.6 Error Handling in Interpreter
- If `json.JSONDecodeError`: raise `InterpreterError("LLM returned non-JSON output")`
- If response is not a list: raise `InterpreterError("LLM returned non-array")`
- If Groq API raises any exception: re-raise as `InterpreterError(str(e))`
- The caller (`main.py`) catches `InterpreterError` and returns HTTP 500 with `{"error": "LLM interpretation failed", "detail": str(e)}`

---

## 7. Guardrail Validator (`guardrails.py`)

### 7.1 Purpose
Deterministically validate every field of the raw LLM output. Reject anything invalid. Never silently invent or patch constraints.

### 7.2 Allowed Directive Types
```python
ALLOWED_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}
```

### 7.3 Complete Validation Rules

Run these checks in order. On first failure, raise `GuardrailError` with a descriptive message.

**A. Count check**
- Length of raw list must equal `len(operator_notes)`
- Raise: `"LLM returned N interpretations for M notes"`

**B. Per-entry checks (for each entry at index i):**

1. `note_index` must equal `i` (entries must be in order 0..N-1)
2. `directive_type` must be in `ALLOWED_DIRECTIVE_TYPES`
3. `applies` type must be boolean

**C. no_op rules:**
- If `directive_type == "no_op"`: `applies` must be `False`
- If `directive_type == "no_op"`: `structured_adjustment` must be `None` or `null`

**D. Non-no_op rules:**
- If `directive_type != "no_op"`: `applies` must be `True`
- If `directive_type != "no_op"`: `structured_adjustment` must not be `None`

**E. Hours validation (for all non-no_op):**
- `structured_adjustment["hours"]` must exist and be a list
- Each hour must be an integer
- Each hour must be in range 0–23
- Hours must be unique
- Hours must be in ascending order
- Raise specific message identifying which check failed

**F. Directive-specific numeric checks:**

`solar_reduction`:
- `factor` must exist in `structured_adjustment`
- `factor` must be a number (int or float)
- `0.0 <= factor <= 1.0`

`minimum_battery_reserve`:
- `minimum_energy_kwh` must exist in `structured_adjustment`
- Must be a finite non-negative number
- Must not exceed `battery.capacity_kwh`

`max_grid_window`:
- `max_grid_kwh` must exist in `structured_adjustment`
- Must be a finite non-negative number

`no_charge_window`, `no_discharge_window`:
- Only `hours` is required in `structured_adjustment`
- No additional numeric fields required

### 7.4 Return Value
On success, return a list of validated directive dicts (same structure, now trusted).

On failure, raise `GuardrailError(message: str)`.

The caller (`main.py`) catches `GuardrailError` and returns HTTP 500 with `{"error": "Guardrail validation failed", "detail": str(e)}`.

---

## 8. LP Optimizer (`optimizer.py`)

### 8.1 Purpose
Given validated directives + 24-hour scenario data + battery config, build and solve a linear program that minimizes total grid electricity cost while satisfying all constraints.

### 8.2 Pre-processing: Apply solar_reduction

Before building the LP:
```python
effective_solar = [h.solar_kwh for h in hours]  # copy of original
for directive in validated_directives:
    if directive["directive_type"] == "solar_reduction":
        factor = directive["structured_adjustment"]["factor"]
        for h in directive["structured_adjustment"]["hours"]:
            effective_solar[h] = hours[h].solar_kwh * factor
```

### 8.3 Pre-processing: Build per-hour constraint sets

```python
# For each hour 0-23:
no_charge_hours = set()
no_discharge_hours = set()
min_reserve_override = {}   # hour -> minimum_energy_kwh (take max with base)
max_grid_override = {}      # hour -> max_grid_kwh

for directive in validated_directives:
    dt = directive["directive_type"]
    adj = directive["structured_adjustment"]
    if dt == "no_charge_window":
        no_charge_hours.update(adj["hours"])
    elif dt == "no_discharge_window":
        no_discharge_hours.update(adj["hours"])
    elif dt == "minimum_battery_reserve":
        for h in adj["hours"]:
            existing = min_reserve_override.get(h, battery.minimum_energy_kwh)
            min_reserve_override[h] = max(existing, adj["minimum_energy_kwh"])
    elif dt == "max_grid_window":
        for h in adj["hours"]:
            existing = max_grid_override.get(h, float('inf'))
            max_grid_override[h] = min(existing, adj["max_grid_kwh"])
```

### 8.4 LP Model Definition

```python
import pulp

prob = pulp.LpProblem("GridWise", pulp.LpMinimize)

# Decision variables
grid        = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(24)]
solar_used  = [pulp.LpVariable(f"solar_{h}", lowBound=0) for h in range(24)]
charge      = [pulp.LpVariable(f"charge_{h}", lowBound=0) for h in range(24)]
discharge   = [pulp.LpVariable(f"discharge_{h}", lowBound=0) for h in range(24)]
batt_energy = [pulp.LpVariable(f"batt_{h}", lowBound=0) for h in range(24)]
# Binary mutual exclusion: cannot charge and discharge in same hour
is_charge   = [pulp.LpVariable(f"ic_{h}", cat="Binary") for h in range(24)]
is_discharge= [pulp.LpVariable(f"id_{h}", cat="Binary") for h in range(24)]
```

### 8.5 Objective Function

```python
prob += pulp.lpSum(
    grid[h] * hours[h].tariff_bdt_per_kwh for h in range(24)
)
```

### 8.6 Constraints

For every hour h in 0..23:

**Energy balance:**
```
grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h]
```

**Solar usage limit:**
```
solar_used[h] <= effective_solar[h]
```

**Battery state transition:**
```
# h == 0:
batt_energy[0] == battery.initial_energy_kwh + charge[0] - discharge[0]
# h > 0:
batt_energy[h] == batt_energy[h-1] + charge[h] - discharge[h]
```

**Battery capacity bounds:**
```
batt_energy[h] >= active_min_reserve[h]   # see below
batt_energy[h] <= battery.capacity_kwh
```
Where:
```python
active_min_reserve[h] = min_reserve_override.get(h, battery.minimum_energy_kwh)
```

**Battery rate limits:**
```
charge[h]    <= battery.max_charge_kwh_per_hour
discharge[h] <= battery.max_discharge_kwh_per_hour
```

**Mutual exclusion (cannot charge and discharge simultaneously):**
```
charge[h]    <= battery.max_charge_kwh_per_hour    * is_charge[h]
discharge[h] <= battery.max_discharge_kwh_per_hour * is_discharge[h]
is_charge[h] + is_discharge[h] <= 1
```

**no_charge_window directive:**
```python
if h in no_charge_hours:
    prob += charge[h] == 0
```

**no_discharge_window directive:**
```python
if h in no_discharge_hours:
    prob += discharge[h] == 0
```

**max_grid_window directive:**
```python
if h in max_grid_override:
    prob += grid[h] <= max_grid_override[h]
```

**End-of-day battery neutrality (hard constraint):**
```
batt_energy[23] == battery.initial_energy_kwh
```

### 8.7 Solving

```python
solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=25)
status = prob.solve(solver)

if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
    raise OptimizerError(f"LP solver returned status: {pulp.LpStatus[status]}")
```

### 8.8 Extracting the Schedule

```python
hourly_plan = []
for h in range(24):
    g   = max(0.0, pulp.value(grid[h]))
    s   = max(0.0, pulp.value(solar_used[h]))
    c   = max(0.0, pulp.value(charge[h]))
    d   = max(0.0, pulp.value(discharge[h]))
    be  = max(0.0, pulp.value(batt_energy[h]))

    # Determine battery_action
    if c > 1e-6:
        action = "charge"
        batt_kwh = round(c, 6)
    elif d > 1e-6:
        action = "discharge"
        batt_kwh = round(d, 6)
    else:
        action = "idle"
        batt_kwh = 0.0

    hourly_plan.append({
        "hour": h,
        "grid_kwh": round(g, 6),
        "solar_used_kwh": round(s, 6),
        "battery_action": action,
        "battery_kwh": round(batt_kwh, 6),
        "battery_energy_after_kwh": round(be, 6),
    })
```

### 8.9 Computing Totals

```python
total_grid_kwh  = round(sum(e["grid_kwh"] for e in hourly_plan), 6)
total_cost_bdt  = round(sum(
    e["grid_kwh"] * hours[e["hour"]].tariff_bdt_per_kwh for e in hourly_plan
), 6)
peak_grid_kwh   = round(max(e["grid_kwh"] for e in hourly_plan), 6)
```

---

## 9. Replay Validator (`replay.py`)

### 9.1 Purpose
After optimization, independently replay the schedule and verify everything is correct. This is what the judge does — do it yourself first.

### 9.2 Tolerance
`TOLERANCE = 0.01`  (kWh and BDT)

### 9.3 Checks to Run

For every hour h in the returned `hourly_plan` (sorted by hour):

**A. Energy balance:**
```
|grid[h] + solar_used[h] + battery_discharge[h] - demand[h] - battery_charge[h]| <= TOLERANCE
```
Where `battery_discharge[h]` = `battery_kwh` if `battery_action == "discharge"` else 0
And   `battery_charge[h]`    = `battery_kwh` if `battery_action == "charge"` else 0

**B. Solar usage limit:**
```
solar_used[h] <= effective_solar[h] + TOLERANCE
```

**C. Non-negative values:**
```
grid_kwh >= -TOLERANCE
solar_used_kwh >= -TOLERANCE
battery_kwh >= -TOLERANCE
battery_energy_after_kwh >= -TOLERANCE
```

**D. Battery idle consistency:**
```
if battery_action == "idle": battery_kwh <= TOLERANCE
```

**E. Battery state transition:**
```
replayed_energy[0] = battery.initial_energy_kwh + charge[0] - discharge[0]
replayed_energy[h] = replayed_energy[h-1] + charge[h] - discharge[h]
|replayed_energy[h] - battery_energy_after_kwh[h]| <= TOLERANCE
```

**F. Battery capacity bounds:**
```
battery_energy_after_kwh[h] >= active_min_reserve[h] - TOLERANCE
battery_energy_after_kwh[h] <= battery.capacity_kwh + TOLERANCE
```

**G. Rate limits:**
```
charge[h]    <= battery.max_charge_kwh_per_hour + TOLERANCE
discharge[h] <= battery.max_discharge_kwh_per_hour + TOLERANCE
```

**H. Mutual exclusion (at most one of charge/discharge nonzero):**
```
NOT (charge[h] > TOLERANCE AND discharge[h] > TOLERANCE)
```

**I. Directive compliance:**
```
if h in no_charge_hours:    charge[h] <= TOLERANCE
if h in no_discharge_hours: discharge[h] <= TOLERANCE
if h in max_grid_override:  grid[h] <= max_grid_override[h] + TOLERANCE
```

**J. End-of-day neutrality:**
```
|battery_energy_after_kwh[23] - battery.initial_energy_kwh| <= TOLERANCE
```

**K. Totals match:**
```
|sum(grid) - total_grid_kwh| <= TOLERANCE
|sum(grid * tariff) - total_cost_bdt| <= TOLERANCE
|max(grid) - peak_grid_kwh| <= TOLERANCE
```

### 9.4 On Failure
Raise `ReplayError(f"Hour {h}: {check_name} failed — {details}")`.
Caller returns HTTP 500 with `{"error": "Schedule validation failed", "detail": str(e)}`.

---

## 10. Main Application (`main.py`)

### 10.1 FastAPI App Setup

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn, os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="GridWise LLM API")
```

### 10.2 GET /health

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

### 10.3 POST /optimize-energy

```python
@app.post("/optimize-energy")
def optimize_energy(request: OptimizeRequest):
    try:
        # Step 1: Call LLM
        raw_interpretations = call_llm(request.operator_notes, request.battery)
    except InterpreterError as e:
        return JSONResponse(status_code=500, content={"error": "LLM interpretation failed", "detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Unexpected interpreter error", "detail": str(e)})

    try:
        # Step 2: Guardrails
        validated = validate_directives(raw_interpretations, request.operator_notes, request.battery)
    except GuardrailError as e:
        return JSONResponse(status_code=500, content={"error": "Guardrail validation failed", "detail": str(e)})

    try:
        # Step 3: Optimize
        hourly_plan, total_grid_kwh, total_cost_bdt, peak_grid_kwh = optimize(
            request.hours, request.battery, validated
        )
    except OptimizerError as e:
        return JSONResponse(status_code=500, content={"error": "Optimization failed", "detail": str(e)})

    try:
        # Step 4: Replay validation
        replay_validate(hourly_plan, request.hours, request.battery, validated,
                        total_grid_kwh, total_cost_bdt, peak_grid_kwh)
    except ReplayError as e:
        return JSONResponse(status_code=500, content={"error": "Schedule validation failed", "detail": str(e)})

    # Step 5: Build response
    directive_interpretation = [
        {
            "note_index": d["note_index"],
            "applies": d["applies"],
            "directive_type": d["directive_type"],
            "structured_adjustment": d["structured_adjustment"],
            "explanation": d["explanation"],
        }
        for d in validated
    ]

    plan_summary = build_plan_summary(validated, total_cost_bdt)

    return {
        "scenario_id": request.scenario_id,
        "directive_interpretation": directive_interpretation,
        "hourly_plan": hourly_plan,
        "total_grid_kwh": total_grid_kwh,
        "total_cost_bdt": total_cost_bdt,
        "peak_grid_kwh": peak_grid_kwh,
        "plan_summary": plan_summary,
    }
```

### 10.4 Global Exception Handler

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
        # Never include exc details, stack traces, or secrets in production
    )
```

### 10.5 Malformed JSON Handler

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid request", "detail": str(exc)}
    )
```

### 10.6 plan_summary Helper

```python
def build_plan_summary(validated_directives: List[dict], total_cost_bdt: float) -> str:
    parts = []
    for d in validated_directives:
        if d["applies"]:
            dt = d["directive_type"]
            adj = d["structured_adjustment"]
            if dt == "solar_reduction":
                parts.append(f"solar reduced to {adj['factor']*100:.0f}% during hours {adj['hours']}")
            elif dt == "minimum_battery_reserve":
                parts.append(f"battery reserve >= {adj['minimum_energy_kwh']} kWh during hours {adj['hours']}")
            elif dt == "no_charge_window":
                parts.append(f"no charging during hours {adj['hours']}")
            elif dt == "no_discharge_window":
                parts.append(f"no discharging during hours {adj['hours']}")
            elif dt == "max_grid_window":
                parts.append(f"grid capped at {adj['max_grid_kwh']} kWh during hours {adj['hours']}")
    applied = "; ".join(parts) if parts else "no operator directives applied"
    return f"Optimized 24-hour schedule with {applied}. Total grid cost: {total_cost_bdt:.2f} BDT."
```

### 10.7 Entry Point

```python
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
```

---

## 11. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Never bake secrets into the image
# Pass GROQ_API_KEY at runtime via -e flag

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t gridwise:latest .
docker run -e GROQ_API_KEY=your_key_here -p 8000:8000 gridwise:latest
```

---

## 12. README.md (required content)

The README must contain all of the following sections:

### Problem
Brief description of the GridWise challenge.

### Architecture
```
Request → FastAPI → Pydantic Validation → Groq LLM (llama3-70b-8192) → 
Guardrail Validator → PuLP LP Optimizer (CBC) → Replay Validator → Response
```

### Setup (local)
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

### Environment Variables
| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key |
| `GROQ_MODEL` | No | `llama3-70b-8192` | Groq model to use |
| `PORT` | No | `8000` | Service port |

### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Sample Request
```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Docker
```bash
docker pull <registry>/<image>:<tag>
docker run -e GROQ_API_KEY=your_key_here -p 8000:8000 <registry>/<image>:<tag>
curl http://localhost:8000/health
```

### LLM Role
Groq LLM (`llama3-70b-8192`) interprets each operator note into a structured directive at `temperature=0`. Output is validated deterministically by guardrails before being applied to the LP optimizer.

### Optimizer
PuLP with the bundled CBC MILP solver. Binary variables enforce mutual exclusion of charge/discharge.

### Guardrails
All LLM output is treated as untrusted. Deterministic validation checks types, ranges, hour uniqueness, and directive-specific numeric constraints before any LLM output touches the optimizer.

### Known Limitations
- Groq API must be reachable during judging; ensure valid API key and quota
- LP solve time is bounded to 25 seconds; scenarios with many tight constraints may approach this limit
- Numeric values use floating-point; differences within 0.01 kWh/BDT are treated as equivalent per spec

---

## 13. Directive Interpretation — Detailed Rules & Edge Cases

These are the exact semantic rules the LLM prompt must enforce and the guardrails must verify.

### 13.1 Time window parsing
| Natural language | Hours array |
|---|---|
| "1 PM to 3 PM" | [13, 14] |
| "2 AM until 5 AM" | [2, 3, 4] |
| "noon until 2 PM" | [12, 13] |
| "6 PM until 9 PM" | [18, 19, 20] |
| "6 PM until 10 PM" | [18, 19, 20, 21] |
| "7 PM until 9 PM" | [19, 20] |
| "10 AM until noon" | [10, 11] |
| "11 AM to 2 PM" | [11, 12, 13] |
| "from 6 to 8 PM" | [18, 19] |
| "from 13:00 to 15:00" | [13, 14] |

Rule: **start included, end excluded**.

### 13.2 solar_reduction factor
| Note wording | factor |
|---|---|
| "drop to about 20%" | 0.2 |
| "roughly 25% of the forecast" | 0.25 |
| "80% reduction" | 0.2 |
| "half of forecast solar" | 0.5 |
| "one-fifth of normal output" | 0.2 |
| "about 50% cloud cover" | 0.5 |

Rule: factor = **remaining fraction**, not the reduction. "80% reduction" means 20% remains = 0.2.

### 13.3 minimum_battery_reserve kwh value
| Note wording | Resolution |
|---|---|
| "at least 120 kWh" | 120.0 |
| "at least 90 kWh" | 90.0 |
| "at least 80 kWh" | 80.0 |
| "50% of battery capacity" (capacity=200) | 100.0 |
| "at least 50% of ... capacity" (capacity=200) | 100.0 |

Rule: if percentage given, multiply by `battery.capacity_kwh`. Include `capacity_kwh` in the LLM prompt context.

### 13.4 no_op classification
A note is `no_op` if it describes anything not directly constraining today's 24-hour energy schedule:
- Administrative events (registration deadlines, library hours, cafeteria menus, seminar bookings)
- Future events ("next week", "next month")
- Events not related to grid, solar, or battery operation
- Notifications about staff or office changes

### 13.5 Multiple directives in one scenario
When multiple notes are provided, each is interpreted independently. The optimizer receives all validated directives and applies all of them simultaneously. Directives from different notes can affect overlapping hours — apply both constraints.

---

## 14. Energy Model — Complete Reference

### 14.1 Effective solar after directive
```
if solar_reduction applies to hour h:
    effective_solar[h] = original_solar[h] * factor
else:
    effective_solar[h] = original_solar[h]
```

### 14.2 Energy balance (must hold every hour, within 0.01 tolerance)
```
grid_kwh[h] + solar_used_kwh[h] + battery_discharge_kwh[h]
  = demand_kwh[h] + battery_charge_kwh[h]
```

Where:
- `battery_discharge_kwh[h]` = `battery_kwh` if `battery_action == "discharge"` else 0
- `battery_charge_kwh[h]`    = `battery_kwh` if `battery_action == "charge"` else 0
- `battery_kwh` must be 0 when `battery_action == "idle"`

### 14.3 Battery state
```
battery_energy_after_kwh[0] = initial_energy_kwh + charge[0] - discharge[0]
battery_energy_after_kwh[h] = battery_energy_after_kwh[h-1] + charge[h] - discharge[h]
```

### 14.4 Battery bounds (every hour)
```
active_minimum[h] <= battery_energy_after_kwh[h] <= capacity_kwh
```
Where:
```
active_minimum[h] = max(battery.minimum_energy_kwh, directive_minimum_energy_kwh[h])
```
(If no `minimum_battery_reserve` directive applies to hour h, use `battery.minimum_energy_kwh`.)

### 14.5 Rate limits (every hour)
```
0 <= charge[h]    <= max_charge_kwh_per_hour
0 <= discharge[h] <= max_discharge_kwh_per_hour
```

### 14.6 Mutual exclusion
```
NOT (charge[h] > 0 AND discharge[h] > 0)
```
Enforced via binary variables in the LP.

### 14.7 Solar usage
```
0 <= solar_used_kwh[h] <= effective_solar[h]
```
Unused solar is curtailed. No grid export.

### 14.8 End-of-day neutrality (hard constraint)
```
battery_energy_after_kwh[23] = battery.initial_energy_kwh
```

### 14.9 Optimization objective
```
minimize: SUM(grid_kwh[h] * tariff_bdt_per_kwh[h]) for h = 0..23
```
Lower cost is better. A lower-cost schedule that violates any constraint is invalid.

---

## 15. Security Requirements

- **Never** commit `.env`, API keys, tokens, or passwords to the repository
- **Never** expose secrets, raw prompts, stack traces, or sensitive config in API responses
- **Never** expose secrets in logs
- All error responses must be controlled and generic
- Use only synthetic scenario data from the harness; do not use real campus, utility, or personal data
- Docker image must not contain baked-in credentials

`.gitignore` must include at minimum:
```
.env
__pycache__/
*.pyc
*.pyo
venv/
.venv/
```

---

## 16. Performance Requirements

| Metric | Requirement |
|---|---|
| `GET /health` response time | < 60s after startup |
| `POST /optimize-energy` timeout | Must complete within 30s |
| p95 latency target | <= 5s for full 3/3 latency points |
| Failure rate | Valid requests must not return 5xx, invalid JSON, or no response |
| Repeated requests | Service must remain stable across multiple back-to-back hidden test requests |
| Malformed input | Must return 400, not crash |
| LLM/provider failure | Must return controlled 500, not crash |

**LP solver time limit:** Set PuLP CBC solver `timeLimit=25` to stay within the 30s HTTP timeout.

---

## 17. Scoring Alignment

This implementation targets the following scoring categories:

| Category | Points | How this spec covers it |
|---|---|---|
| LLM Directive Interpretation | 25 | Structured prompt with exact rules, `temperature=0`, percentage→kWh resolution, paraphrase-robust |
| Directive Application & Constraint Correctness | 25 | LP hard constraints for every directive type, replay validation |
| Optimization Quality | 10 | PuLP minimizes cost; binary mutual exclusion; proper effective solar pre-processing |
| API Contract & Schema | 10 | Pydantic models match exact spec fields; 400/500 handling; schema validation |
| Performance & Reliability | 10 | 25s LP time limit; controlled error handling; no crashes on bad input |
| Deployment & Docker Fallback | 10 | Dockerfile provided; binds to 0.0.0.0; no baked-in secrets |
| Documentation & Local Reproducibility | 10 | README with copy-paste quickstart, env var table, curl examples, Docker pull/run |

---

## 18. Full Example: Expected Request → Response (SAMPLE-01)

### Request
```json
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [
    {"hour": 0,  "demand_kwh": 90,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
    {"hour": 1,  "demand_kwh": 85,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
    {"hour": 2,  "demand_kwh": 80,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
    {"hour": 3,  "demand_kwh": 80,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
    {"hour": 4,  "demand_kwh": 85,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
    {"hour": 5,  "demand_kwh": 95,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
    {"hour": 6,  "demand_kwh": 110, "solar_kwh": 5,   "tariff_bdt_per_kwh": 8},
    {"hour": 7,  "demand_kwh": 130, "solar_kwh": 20,  "tariff_bdt_per_kwh": 10},
    {"hour": 8,  "demand_kwh": 150, "solar_kwh": 50,  "tariff_bdt_per_kwh": 12},
    {"hour": 9,  "demand_kwh": 165, "solar_kwh": 90,  "tariff_bdt_per_kwh": 14},
    {"hour": 10, "demand_kwh": 175, "solar_kwh": 130, "tariff_bdt_per_kwh": 16},
    {"hour": 11, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
    {"hour": 12, "demand_kwh": 185, "solar_kwh": 180, "tariff_bdt_per_kwh": 15},
    {"hour": 13, "demand_kwh": 180, "solar_kwh": 170, "tariff_bdt_per_kwh": 14},
    {"hour": 14, "demand_kwh": 170, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
    {"hour": 15, "demand_kwh": 165, "solar_kwh": 90,  "tariff_bdt_per_kwh": 14},
    {"hour": 16, "demand_kwh": 170, "solar_kwh": 45,  "tariff_bdt_per_kwh": 18},
    {"hour": 17, "demand_kwh": 185, "solar_kwh": 10,  "tariff_bdt_per_kwh": 22},
    {"hour": 18, "demand_kwh": 205, "solar_kwh": 0,   "tariff_bdt_per_kwh": 28},
    {"hour": 19, "demand_kwh": 215, "solar_kwh": 0,   "tariff_bdt_per_kwh": 30},
    {"hour": 20, "demand_kwh": 205, "solar_kwh": 0,   "tariff_bdt_per_kwh": 26},
    {"hour": 21, "demand_kwh": 175, "solar_kwh": 0,   "tariff_bdt_per_kwh": 18},
    {"hour": 22, "demand_kwh": 135, "solar_kwh": 0,   "tariff_bdt_per_kwh": 10},
    {"hour": 23, "demand_kwh": 105, "solar_kwh": 0,   "tariff_bdt_per_kwh": 7}
  ],
  "battery": {
    "capacity_kwh": 220,
    "initial_energy_kwh": 110,
    "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 50
  }
}
```

### Expected directive_interpretation
```json
[
  {
    "note_index": 0,
    "applies": true,
    "directive_type": "solar_reduction",
    "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
    "explanation": "Solar availability is reduced to 25% during the panel-cleaning window (noon to 2 PM)."
  },
  {
    "note_index": 1,
    "applies": false,
    "directive_type": "no_op",
    "structured_adjustment": null,
    "explanation": "The sports office note does not affect the 24-hour energy schedule."
  }
]
```

Note:
- "noon until 2 PM" → hours [12, 13] (start inclusive, end exclusive)
- "roughly 25% of the forecast" → factor = 0.25 (remaining fraction)
- Second note is administrative → no_op

---

## 19. Critical Failure Modes to Avoid

| Failure | Cause | Prevention |
|---|---|---|
| Wrong hours for "2 AM until 5 AM" | Off-by-one (returning [2,3,4,5]) | Prompt: end hour excluded; "until 5 AM" = stop before 5 |
| Wrong factor for "80% reduction" | Returning factor=0.8 instead of 0.2 | Prompt: factor = remaining fraction; 80% reduction → 0.2 |
| Crashing on malformed LLM JSON | No try/except around json.loads | Always wrap in try/except, return 500 |
| Crashing on LP infeasible | No status check | Always check `pulp.LpStatus[status]` |
| Battery not returning to initial level | Missing end-of-day constraint | Hard constraint: `batt_energy[23] == initial_energy_kwh` |
| Simultaneous charge and discharge | No mutual exclusion | Binary variables + constraint `is_charge + is_discharge <= 1` |
| Secrets in response | Passing `str(exception)` with raw stack trace | Return only generic error messages in production |
| total_cost_bdt mismatch | Using approximate values | Recalculate from hourly_plan in replay validator |
| Inventing unsupported directive type | LLM hallucination | Guardrail check against `ALLOWED_DIRECTIVE_TYPES` set |
| Solar used exceeds effective solar | Missing post-reduction solar cap | Guardrail + LP constraint: `solar_used[h] <= effective_solar[h]` |

---

## 20. Repository Policy

- Create a **new GitHub repository** after the question is revealed
- Keep **private** during the event window (7 PM – 11 PM)
- Make **public** after the submission deadline
- Submitted public URL must remain reachable during the entire evaluation window
- All required submission links (Docker image, video) must remain accessible through the judging window
