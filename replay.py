from typing import List

from models import BatteryConfig, HourEntry


class ReplayError(Exception):
    pass


TOLERANCE = 0.01


def replay_validate(
    hourly_plan: List[dict],
    hours: List[HourEntry],
    battery: BatteryConfig,
    validated_directives: List[dict],
    total_grid_kwh: float,
    total_cost_bdt: float,
    peak_grid_kwh: float,
) -> None:
    hours_sorted = sorted(hours, key=lambda h: h.hour)
    plan = sorted(hourly_plan, key=lambda e: e["hour"])
    demand = {h.hour: h.demand_kwh for h in hours_sorted}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours_sorted}

    # Recompute effective solar + constraint sets (mirror optimizer)
    effective_solar = [h.solar_kwh for h in hours_sorted]
    no_charge_hours = set()
    no_discharge_hours = set()
    min_reserve_override = {}
    max_grid_override = {}
    for directive in validated_directives:
        dt = directive["directive_type"]
        if dt == "no_op":
            continue
        adj = directive["structured_adjustment"]
        if dt == "solar_reduction":
            for h in adj["hours"]:
                effective_solar[h] = hours_sorted[h].solar_kwh * adj["factor"]
        elif dt == "no_charge_window":
            no_charge_hours.update(adj["hours"])
        elif dt == "no_discharge_window":
            no_discharge_hours.update(adj["hours"])
        elif dt == "minimum_battery_reserve":
            for h in adj["hours"]:
                existing = min_reserve_override.get(h, battery.minimum_energy_kwh)
                min_reserve_override[h] = max(existing, adj["minimum_energy_kwh"])
        elif dt == "max_grid_window":
            for h in adj["hours"]:
                existing = max_grid_override.get(h, float("inf"))
                max_grid_override[h] = min(existing, adj["max_grid_kwh"])

    def fail(h, check, details):
        raise ReplayError(f"Hour {h}: {check} failed — {details}")

    replayed_energy = {}
    for e in plan:
        h = e["hour"]
        g = e["grid_kwh"]
        s = e["solar_used_kwh"]
        action = e["battery_action"]
        bkwh = e["battery_kwh"]
        be = e["battery_energy_after_kwh"]

        charge = bkwh if action == "charge" else 0.0
        discharge = bkwh if action == "discharge" else 0.0

        # A. Energy balance
        if abs(g + s + discharge - demand[h] - charge) > TOLERANCE:
            fail(h, "energy balance", f"{g}+{s}+{discharge} vs {demand[h]}+{charge}")

        # B. Solar usage limit
        if s > effective_solar[h] + TOLERANCE:
            fail(h, "solar usage", f"{s} > effective {effective_solar[h]}")

        # C. Non-negative values
        if g < -TOLERANCE or s < -TOLERANCE or bkwh < -TOLERANCE or be < -TOLERANCE:
            fail(h, "non-negative", f"grid={g} solar={s} bkwh={bkwh} be={be}")

        # D. Idle consistency
        if action == "idle" and bkwh > TOLERANCE:
            fail(h, "idle consistency", f"idle with battery_kwh={bkwh}")
        if action not in ("charge", "discharge", "idle"):
            fail(h, "battery_action", f"invalid action '{action}'")

        # E. State transition
        if h == 0:
            replayed = battery.initial_energy_kwh + charge - discharge
        else:
            replayed = replayed_energy[h - 1] + charge - discharge
        replayed_energy[h] = replayed
        if abs(replayed - be) > TOLERANCE:
            fail(h, "state transition", f"replayed={replayed} vs reported={be}")

        # F. Capacity bounds
        active_min = min_reserve_override.get(h, battery.minimum_energy_kwh)
        if be < active_min - TOLERANCE:
            fail(h, "min reserve", f"{be} < {active_min}")
        if be > battery.capacity_kwh + TOLERANCE:
            fail(h, "capacity", f"{be} > {battery.capacity_kwh}")

        # G. Rate limits
        if charge > battery.max_charge_kwh_per_hour + TOLERANCE:
            fail(h, "charge rate", f"{charge} > {battery.max_charge_kwh_per_hour}")
        if discharge > battery.max_discharge_kwh_per_hour + TOLERANCE:
            fail(h, "discharge rate", f"{discharge} > {battery.max_discharge_kwh_per_hour}")

        # H. Mutual exclusion
        if charge > TOLERANCE and discharge > TOLERANCE:
            fail(h, "mutual exclusion", f"charge={charge} discharge={discharge}")

        # I. Directive compliance
        if h in no_charge_hours and charge > TOLERANCE:
            fail(h, "no_charge_window", f"charge={charge}")
        if h in no_discharge_hours and discharge > TOLERANCE:
            fail(h, "no_discharge_window", f"discharge={discharge}")
        if h in max_grid_override and g > max_grid_override[h] + TOLERANCE:
            fail(h, "max_grid_window", f"grid={g} > {max_grid_override[h]}")

    # J. End-of-day neutrality
    if abs(plan[-1]["battery_energy_after_kwh"] - battery.initial_energy_kwh) > TOLERANCE:
        fail(23, "end-of-day neutrality",
             f"{plan[-1]['battery_energy_after_kwh']} vs {battery.initial_energy_kwh}")

    # K. Totals match
    grid_sum = sum(e["grid_kwh"] for e in plan)
    cost_sum = sum(e["grid_kwh"] * tariff[e["hour"]] for e in plan)
    peak = max(e["grid_kwh"] for e in plan)
    if abs(grid_sum - total_grid_kwh) > TOLERANCE:
        raise ReplayError(
            f"Totals mismatch: sum(grid)={grid_sum} vs total_grid_kwh={total_grid_kwh}"
        )
    if abs(cost_sum - total_cost_bdt) > TOLERANCE:
        raise ReplayError(
            f"Totals mismatch: sum(cost)={cost_sum} vs total_cost_bdt={total_cost_bdt}"
        )
    if abs(peak - peak_grid_kwh) > TOLERANCE:
        raise ReplayError(
            f"Totals mismatch: max(grid)={peak} vs peak_grid_kwh={peak_grid_kwh}"
        )
