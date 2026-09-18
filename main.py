import logging
import os
from pathlib import Path
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from guardrails import GuardrailError, validate_directives
from interpreter import InterpreterError, call_llm
from models import OptimizeRequest
from optimizer import OptimizerError, optimize
from replay import ReplayError, replay_validate

# override=True makes the repo's .env authoritative. Otherwise a stale/externally
# exported GROQ_MODEL (e.g. a decommissioned model id) silently overrides the
# model we actually intend to use and the whole service returns 500s.
load_dotenv(override=True)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="GridWise LLM API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def dashboard():
    # Static demo console. Additive only: judge endpoints below are unchanged.
    return FileResponse(
        Path(__file__).with_name("static").joinpath("index.html"),
        media_type="text/html",
    )


@app.post("/optimize-energy")
def optimize_energy(request: OptimizeRequest):
    try:
        raw_interpretations = call_llm(request.operator_notes, request.battery)
    except InterpreterError as e:
        logger.error("LLM interpretation failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "LLM interpretation failed"},
        )
    except Exception as e:
        logger.error("Unexpected interpreter error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected interpreter error"},
        )

    try:
        validated = validate_directives(
            raw_interpretations, request.operator_notes, request.battery
        )
    except GuardrailError as e:
        logger.error("Guardrail validation failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Guardrail validation failed"},
        )

    try:
        hourly_plan, total_grid_kwh, total_cost_bdt, peak_grid_kwh = optimize(
            request.hours, request.battery, validated
        )
    except OptimizerError as e:
        logger.error("Optimization failed: %s", e)
        return JSONResponse(
            status_code=500, content={"error": "Optimization failed"}
        )

    try:
        replay_validate(
            hourly_plan,
            request.hours,
            request.battery,
            validated,
            total_grid_kwh,
            total_cost_bdt,
            peak_grid_kwh,
        )
    except ReplayError as e:
        logger.error("Schedule validation failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Schedule validation failed"},
        )

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


def build_plan_summary(validated_directives: List[dict], total_cost_bdt: float) -> str:
    parts = []
    for d in validated_directives:
        if d["applies"]:
            dt = d["directive_type"]
            adj = d["structured_adjustment"]
            if dt == "solar_reduction":
                parts.append(
                    f"solar reduced to {adj['factor'] * 100:.0f}% during hours {adj['hours']}"
                )
            elif dt == "minimum_battery_reserve":
                parts.append(
                    f"battery reserve >= {adj['minimum_energy_kwh']} kWh during hours {adj['hours']}"
                )
            elif dt == "no_charge_window":
                parts.append(f"no charging during hours {adj['hours']}")
            elif dt == "no_discharge_window":
                parts.append(f"no discharging during hours {adj['hours']}")
            elif dt == "max_grid_window":
                parts.append(
                    f"grid capped at {adj['max_grid_kwh']} kWh during hours {adj['hours']}"
                )
    applied = "; ".join(parts) if parts else "no operator directives applied"
    return (
        f"Optimized 24-hour schedule with {applied}. "
        f"Total grid cost: {total_cost_bdt:.2f} BDT."
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the full validation detail server-side only. Never echo the raw request
    # body or internal file paths back to the client (see audit finding #4).
    logger.error("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=400, content={"error": "Invalid request"}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
