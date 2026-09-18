import json
import os
from typing import List

from groq import Groq

from models import BatteryConfig


class InterpreterError(Exception):
    pass


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise InterpreterError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def _get_model() -> str:
    return os.getenv("GROQ_MODEL", "llama3-70b-8192")


SYSTEM_PROMPT = """You are an energy management system operator-note interpreter.

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
NEVER output anything outside the JSON array."""


def build_user_prompt(notes: List[str], battery: BatteryConfig) -> str:
    notes_text = "\n".join(f'{i}. "{note}"' for i, note in enumerate(notes))
    return (
        f"Battery capacity for this scenario: {battery.capacity_kwh} kWh\n"
        f"Initial battery energy: {battery.initial_energy_kwh} kWh\n"
        f"\n"
        f"Operator notes to interpret ({len(notes)} total):\n"
        f"{notes_text}\n"
        f"\n"
        f"Return a JSON array with exactly {len(notes)} objects, "
        f"one per note, in note_index order 0 to {len(notes) - 1}."
    )


def call_llm(notes: List[str], battery: BatteryConfig) -> List[dict]:
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(notes, battery)},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()
    except InterpreterError:
        raise
    except Exception as e:
        raise InterpreterError(str(e))

    if raw.startswith("```"):
        parts = raw.split("```")
        # ponytail: fence-strip, handles ```json ... ``` and bare ``` ... ```
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise InterpreterError("LLM returned non-JSON output")

    if not isinstance(parsed, list):
        raise InterpreterError("LLM returned non-array")

    return parsed
