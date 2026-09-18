import os
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from guardrails import GuardrailError, validate_directives
from interpreter import InterpreterError, call_llm
from models import OptimizeRequest
from optimizer import OptimizerError, optimize
from replay import ReplayError, replay_validate

load_dotenv()

app = FastAPI(title="GridWise LLM API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize-energy")
def optimize_energy(request: OptimizeRequest):
    try:
        raw_interpretations = call_llm(request.operator_notes, request.battery)
    except InterpreterError as e:
        return JSONResponse(
            status_code=500,
            content={"error": "LLM interpretation failed", "detail": str(e)},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected interpreter error", "detail": str(e)},
        )

    try:
        validated = validate_directives(
            raw_interpretations, request.operator_notes, request.battery
        )
    except GuardrailError as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Guardrail validation failed", "detail": str(e)},
        )

    try:
        hourly_plan, total_grid_kwh, total_cost_bdt, peak_grid_kwh = optimize(
            request.hours, request.battery, validated
        )
    except OptimizerError as e:
        return JSONResponse(
            status_code=500, content={"error": "Optimization failed", "detail": str(e)}
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
        return JSONResponse(
            status_code=500,
            content={"error": "Schedule validation failed", "detail": str(e)},
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
    return JSONResponse(
        status_code=400, content={"error": "Invalid request", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
