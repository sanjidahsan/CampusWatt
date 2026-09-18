from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class HourEntry(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatteryConfig(BaseModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(gt=0)
    max_discharge_kwh_per_hour: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_levels(self):
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh must be <= capacity_kwh")
        if not self.minimum_energy_kwh < self.capacity_kwh:
            raise ValueError("minimum_energy_kwh must be < capacity_kwh")
        return self


class OptimizeRequest(BaseModel):
    scenario_id: str = Field(min_length=1)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: BatteryConfig

    @field_validator("scenario_id")
    @classmethod
    def validate_scenario_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("scenario_id must be non-empty")
        return v

    @field_validator("operator_notes")
    @classmethod
    def validate_notes(cls, v: List[str]) -> List[str]:
        if not 1 <= len(v) <= 3:
            raise ValueError("operator_notes must have 1-3 items")
        if not all(s.strip() for s in v):
            raise ValueError("operator_notes items must be non-empty")
        return v

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, v: List[HourEntry]) -> List[HourEntry]:
        if len(v) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        hour_vals = [h.hour for h in v]
        if sorted(hour_vals) != list(range(24)):
            raise ValueError("hours must cover 0-23 uniquely")
        return sorted(v, key=lambda h: h.hour)


class StructuredAdjustment(BaseModel):
    hours: Optional[List[int]] = None
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: Optional[StructuredAdjustment] = None
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
