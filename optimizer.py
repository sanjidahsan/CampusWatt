from typing import List, Tuple

import pulp

from models import BatteryConfig, HourEntry


class OptimizerError(Exception):
    pass


def optimize(
    hours: List[HourEntry],
    battery: BatteryConfig,
    validated_directives: List[dict],
) -> Tuple[List[dict], float, float, float]:
    hours = sorted(hours, key=lambda h: h.hour)

    # Pre-processing: effective solar after solar_reduction
    effective_solar = [h.solar_kwh for h in hours]
    for directive in validated_directives:
        if directive["directive_type"] == "solar_reduction":
            factor = directive["structured_adjustment"]["factor"]
            for h in directive["structured_adjustment"]["hours"]:
                effective_solar[h] *= factor  # combine overlapping reductions conservatively

    # Pre-processing: per-hour constraint sets
    no_charge_hours = set()
    no_discharge_hours = set()
    min_reserve_override = {}
    max_grid_override = {}

    for directive in validated_directives:
        dt = directive["directive_type"]
        if dt == "no_op":
            continue
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
                existing = max_grid_override.get(h, float("inf"))
                max_grid_override[h] = min(existing, adj["max_grid_kwh"])

    prob = pulp.LpProblem("GridWise", pulp.LpMinimize)

    grid = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(24)]
    solar_used = [pulp.LpVariable(f"solar_{h}", lowBound=0) for h in range(24)]
    charge = [pulp.LpVariable(f"charge_{h}", lowBound=0) for h in range(24)]
    discharge = [pulp.LpVariable(f"discharge_{h}", lowBound=0) for h in range(24)]
    batt_energy = [pulp.LpVariable(f"batt_{h}", lowBound=0) for h in range(24)]
    is_charge = [pulp.LpVariable(f"ic_{h}", cat="Binary") for h in range(24)]
    is_discharge = [pulp.LpVariable(f"id_{h}", cat="Binary") for h in range(24)]

    prob += pulp.lpSum(
        grid[h] * hours[h].tariff_bdt_per_kwh for h in range(24)
    )

    for h in range(24):
        demand = hours[h].demand_kwh

        # Energy balance
        prob += grid[h] + solar_used[h] + discharge[h] == demand + charge[h]

        # Solar usage limit
        prob += solar_used[h] <= effective_solar[h]

        # Battery state transition
        if h == 0:
            prob += (
                batt_energy[0]
                == battery.initial_energy_kwh + charge[0] - discharge[0]
            )
        else:
            prob += (
                batt_energy[h]
                == batt_energy[h - 1] + charge[h] - discharge[h]
            )

        # Capacity bounds
        active_min = min_reserve_override.get(h, battery.minimum_energy_kwh)
        prob += batt_energy[h] >= active_min
        prob += batt_energy[h] <= battery.capacity_kwh

        # Rate limits + mutual exclusion
        prob += charge[h] <= battery.max_charge_kwh_per_hour * is_charge[h]
        prob += discharge[h] <= battery.max_discharge_kwh_per_hour * is_discharge[h]
        prob += is_charge[h] + is_discharge[h] <= 1

        if h in no_charge_hours:
            prob += charge[h] == 0
        if h in no_discharge_hours:
            prob += discharge[h] == 0
        if h in max_grid_override:
            prob += grid[h] <= max_grid_override[h]

    # End-of-day neutrality
    prob += batt_energy[23] == battery.initial_energy_kwh

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=25)
    try:
        status = prob.solve(solver)
    except Exception as e:
        raise OptimizerError(f"LP solver failed: {e}")

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        raise OptimizerError(
            f"LP solver returned status: {pulp.LpStatus[status]}"
        )

    hourly_plan = []
    for h in range(24):
        g = max(0.0, pulp.value(grid[h]) or 0.0)
        s = max(0.0, pulp.value(solar_used[h]) or 0.0)
        c = max(0.0, pulp.value(charge[h]) or 0.0)
        d = max(0.0, pulp.value(discharge[h]) or 0.0)
        be = max(0.0, pulp.value(batt_energy[h]) or 0.0)

        if c > 1e-6:
            action = "charge"
            batt_kwh = round(c, 6)
        elif d > 1e-6:
            action = "discharge"
            batt_kwh = round(d, 6)
        else:
            action = "idle"
            batt_kwh = 0.0

        hourly_plan.append(
            {
                "hour": h,
                "grid_kwh": round(g, 6),
                "solar_used_kwh": round(s, 6),
                "battery_action": action,
                "battery_kwh": batt_kwh,
                "battery_energy_after_kwh": round(be, 6),
            }
        )

    total_grid_kwh = round(sum(e["grid_kwh"] for e in hourly_plan), 6)
    total_cost_bdt = round(
        sum(
            e["grid_kwh"] * hours[e["hour"]].tariff_bdt_per_kwh
            for e in hourly_plan
        ),
        6,
    )
    peak_grid_kwh = round(max(e["grid_kwh"] for e in hourly_plan), 6)

    return hourly_plan, total_grid_kwh, total_cost_bdt, peak_grid_kwh
