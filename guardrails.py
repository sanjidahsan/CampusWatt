import math
from typing import List

from models import BatteryConfig


class GuardrailError(Exception):
    pass


ALLOWED_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_directives(
    raw: List[dict],
    operator_notes: List[str],
    battery: BatteryConfig,
) -> List[dict]:
    if not isinstance(raw, list):
        raise GuardrailError("LLM output must be a JSON array")
    if len(raw) != len(operator_notes):
        raise GuardrailError(
            f"LLM returned {len(raw)} interpretations for {len(operator_notes)} notes"
        )

    validated = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise GuardrailError(f"Entry {i} must be an object")

        if entry.get("note_index") != i:
            raise GuardrailError(
                f"Entry {i}: note_index must equal {i}"
            )

        dt = entry.get("directive_type")
        if dt not in ALLOWED_DIRECTIVE_TYPES:
            raise GuardrailError(
                f"Entry {i}: invalid directive_type '{dt}'"
            )

        applies = entry.get("applies")
        if not isinstance(applies, bool):
            raise GuardrailError(f"Entry {i}: applies must be boolean")

        adj = entry.get("structured_adjustment")
        explanation = entry.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise GuardrailError(f"Entry {i}: explanation must be non-empty string")

        if dt == "no_op":
            if applies is not False:
                raise GuardrailError(f"Entry {i}: no_op must have applies=false")
            if adj is not None:
                raise GuardrailError(
                    f"Entry {i}: no_op must have structured_adjustment=null"
                )
            validated.append(entry)
            continue

        # Non-no_op rules
        if applies is not True:
            raise GuardrailError(
                f"Entry {i}: directive_type '{dt}' must have applies=true"
            )
        if adj is None or not isinstance(adj, dict):
            raise GuardrailError(
                f"Entry {i}: directive_type '{dt}' must have structured_adjustment object"
            )

        # Hours validation
        hours = adj.get("hours")
        if not isinstance(hours, list) or len(hours) == 0:
            raise GuardrailError(
                f"Entry {i}: structured_adjustment.hours must be a non-empty list"
            )
        for h in hours:
            if not isinstance(h, int) or isinstance(h, bool):
                raise GuardrailError(f"Entry {i}: hours must be integers")
            if not 0 <= h <= 23:
                raise GuardrailError(f"Entry {i}: hour {h} out of range 0-23")
        if len(set(hours)) != len(hours):
            raise GuardrailError(f"Entry {i}: hours must be unique")
        if hours != sorted(hours):
            raise GuardrailError(f"Entry {i}: hours must be in ascending order")

        # Directive-specific numeric checks
        if dt == "solar_reduction":
            factor = adj.get("factor")
            if not _is_number(factor):
                raise GuardrailError(
                    f"Entry {i}: solar_reduction requires numeric factor"
                )
            if not 0.0 <= float(factor) <= 1.0:
                raise GuardrailError(
                    f"Entry {i}: solar_reduction factor must be in [0.0, 1.0]"
                )
        elif dt == "minimum_battery_reserve":
            v = adj.get("minimum_energy_kwh")
            if not _is_number(v):
                raise GuardrailError(
                    f"Entry {i}: minimum_battery_reserve requires numeric minimum_energy_kwh"
                )
            fv = float(v)
            if not math.isfinite(fv) or fv < 0:
                raise GuardrailError(
                    f"Entry {i}: minimum_energy_kwh must be a finite non-negative number"
                )
            if fv > battery.capacity_kwh:
                raise GuardrailError(
                    f"Entry {i}: minimum_energy_kwh must not exceed battery capacity"
                )
        elif dt == "max_grid_window":
            v = adj.get("max_grid_kwh")
            if not _is_number(v):
                raise GuardrailError(
                    f"Entry {i}: max_grid_window requires numeric max_grid_kwh"
                )
            fv = float(v)
            if not math.isfinite(fv) or fv < 0:
                raise GuardrailError(
                    f"Entry {i}: max_grid_kwh must be a finite non-negative number"
                )
        elif dt in ("no_charge_window", "no_discharge_window"):
            pass  # hours-only, no extra numeric fields required

        validated.append(entry)

    return validated
