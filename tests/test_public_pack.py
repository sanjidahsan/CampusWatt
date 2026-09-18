"""Regression tests: public pack replay + guardrail negatives + API contract.

Run with stdlib only:  python -m unittest discover -s tests -v
Requires service deps installed (requirements.txt), no extra test deps.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from models import BatteryConfig, OptimizeRequest
from guardrails import GuardrailError, validate_directives
from interpreter import InterpreterError
from optimizer import optimize
from replay import replay_validate

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "context" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def make_battery(**kw):
    base = dict(
        capacity_kwh=200,
        initial_energy_kwh=100,
        minimum_energy_kwh=20,
        max_charge_kwh_per_hour=50,
        max_discharge_kwh_per_hour=50,
    )
    base.update(kw)
    return BatteryConfig(**base)


class PublicPackTest(unittest.TestCase):
    def test_all_reference_directives_produce_valid_schedules(self):
        data = json.loads(PACK.read_text(encoding="utf-8"))
        self.assertEqual(len(data["cases"]), 10)
        for case in data["cases"]:
            with self.subTest(case=case["id"]):
                req = OptimizeRequest.model_validate(case["input"])
                validated = validate_directives(
                    case["expected_output"]["directive_interpretation"],
                    req.operator_notes,
                    req.battery,
                )
                plan, grid, cost, peak = optimize(req.hours, req.battery, validated)
                replay_validate(
                    plan, req.hours, req.battery, validated, grid, cost, peak
                )
                # Solver must be at least as cheap as the reference optimum.
                self.assertLessEqual(
                    cost, case["expected_output"]["total_cost_bdt"] + 0.01
                )

    def test_overlapping_solar_reductions_combine(self):
        req = OptimizeRequest.model_validate(
            json.loads((ROOT / "sample_request.json").read_text(encoding="utf-8"))
        )
        overlap = [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [12], "factor": 0.5},
                "explanation": "x",
            },
            {
                "note_index": 1,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [12], "factor": 0.5},
                "explanation": "x",
            },
        ]
        validated = validate_directives(overlap, ["a", "b"], req.battery)
        plan, grid, cost, peak = optimize(req.hours, req.battery, validated)
        replay_validate(plan, req.hours, req.battery, validated, grid, cost, peak)
        used = next(e for e in plan if e["hour"] == 12)["solar_used_kwh"]
        self.assertLessEqual(used, req.hours[12].solar_kwh * 0.25 + 0.01)


class GuardrailTest(unittest.TestCase):
    def test_accepts_all_six_types(self):
        batt = make_battery()
        cases = [
            ("solar_reduction", {"hours": [13, 14], "factor": 0.2}),
            ("minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 100}),
            ("no_charge_window", {"hours": [2, 3]}),
            ("no_discharge_window", {"hours": [18]}),
            ("max_grid_window", {"hours": [18], "max_grid_kwh": 155}),
            ("no_op", None),
        ]
        for dt, adj in cases:
            with self.subTest(dt=dt):
                applies = dt != "no_op"
                validate_directives(
                    [
                        {
                            "note_index": 0,
                            "applies": applies,
                            "directive_type": dt,
                            "structured_adjustment": adj,
                            "explanation": "x",
                        }
                    ],
                    ["note"],
                    batt,
                )

    def test_rejects_invalid(self):
        batt = make_battery()
        bad = [
            [
                {
                    "note_index": 1,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": [1]},
                    "explanation": "x",
                }
            ],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "fly",
                    "structured_adjustment": {"hours": [1]},
                    "explanation": "x",
                }
            ],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "x",
                }
            ],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": [1], "factor": 1.5},
                    "explanation": "x",
                }
            ],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": [3, 2]},
                    "explanation": "x",
                }
            ],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": [1]},
                    "explanation": "  ",
                }
            ],
        ]
        for i, raw in enumerate(bad):
            with self.subTest(i=i):
                with self.assertRaises(GuardrailError):
                    validate_directives(raw, ["note"], batt)


class ApiContractTest(unittest.TestCase):
    def test_health_and_validation_codes(self):
        from fastapi.testclient import TestClient
        from main import app

        os.environ.pop("GROQ_API_KEY", None)
        client = TestClient(app, raise_server_exceptions=False)
        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.get("/health").json(), {"status": "ok"})

        sample = json.loads((ROOT / "sample_request.json").read_text(encoding="utf-8"))
        r = client.post("/optimize-energy", json=sample)
        self.assertEqual(r.status_code, 500)  # no key -> controlled LLM failure
        self.assertEqual(r.json()["error"], "LLM interpretation failed")

        short = dict(sample)
        short["hours"] = sample["hours"][:10]
        self.assertEqual(client.post("/optimize-energy", json=short).status_code, 400)

        r = client.post(
            "/optimize-energy",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()


class RateLimitError(Exception):
    pass


def _fake_ok(payload):
    msg = SimpleNamespace(content=json.dumps(payload))
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


class RetryTest(unittest.TestCase):
    def _batt(self):
        return make_battery()

    def test_transient_429_is_retried(self):
        import interpreter

        payload = [{"note_index": 0, "applies": False, "directive_type": "no_op",
                    "structured_adjustment": None, "explanation": "x"}]
        fake = mock.Mock()
        fake.chat.completions.create.side_effect = [
            RateLimitError("Error code: 429 - rate limit reached"),
            _fake_ok(payload),
        ]
        with mock.patch.object(interpreter, "_get_client", return_value=fake), \
                mock.patch.object(interpreter.time, "sleep") as slept:
            out = interpreter.call_llm(["note"], self._batt())
        self.assertEqual(out, payload)
        slept.assert_called_once_with(5.0)
        self.assertEqual(fake.chat.completions.create.call_count, 2)

    def test_persistent_429_fails_after_bounded_retries(self):
        import interpreter

        fake = mock.Mock()
        fake.chat.completions.create.side_effect = RateLimitError("429")
        with mock.patch.object(interpreter, "_get_client", return_value=fake), \
                mock.patch.object(interpreter.time, "sleep") as slept:
            with self.assertRaises(InterpreterError):
                interpreter.call_llm(["note"], self._batt())
        self.assertEqual(fake.chat.completions.create.call_count, 3)
        self.assertEqual([c.args[0] for c in slept.call_args_list], [5.0, 10.0])

    def test_non_rate_limit_error_fails_fast(self):
        import interpreter

        fake = mock.Mock()
        fake.chat.completions.create.side_effect = ValueError("boom")
        with mock.patch.object(interpreter, "_get_client", return_value=fake), \
                mock.patch.object(interpreter.time, "sleep") as slept:
            with self.assertRaises(InterpreterError):
                interpreter.call_llm(["note"], self._batt())
        slept.assert_not_called()
        self.assertEqual(fake.chat.completions.create.call_count, 1)


class DashboardTest(unittest.TestCase):
    def test_root_serves_console_without_touching_api(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers.get("content-type", ""))
        self.assertIn("Dispatch console", r.text)
        # Judge endpoints intact
        self.assertEqual(client.get("/health").json(), {"status": "ok"})
        self.assertEqual(client.get("/nope").status_code, 404)
