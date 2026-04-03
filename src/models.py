"""Pydantic typed contracts for the ATC environment."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class LifecyclePhase(StrEnum):
    """Enum covering all lifecycle phases."""

    APPROACH = "approach"
    LANDING = "landing"
    ARRIVAL_HANDOFF = "arrival_handoff"
    TAXI_IN = "taxi_in"
    DOCKING = "docking"
    AT_GATE = "at_gate"
    PUSHBACK = "pushback"
    TAXI_OUT = "taxi_out"
    DEPARTURE_QUEUE = "departure_queue"
    TAKEOFF = "takeoff"
    DEPARTED = "departed"


class ClearanceType(StrEnum):
    """Standardized clearance types."""

    PUSHBACK = "pushback"
    TAXI = "taxi"
    HOLD_SHORT = "hold_short"
    CROSS_RUNWAY = "cross_runway"
    LINE_UP = "line_up"
    TAKEOFF = "takeoff"
    LANDING = "landing"


class Action(BaseModel):
    """Structured controller command."""

    model_config = {"extra": "forbid"}

    clearance_type: ClearanceType
    target_callsign: str = Field(..., min_length=1, max_length=20)
    route: list[str] = Field(default_factory=list)
    readback_required: bool = False
    pushback_direction: str | None = None
    hold_short: bool = False
    runway: str | None = None


class Observation(BaseModel):
    """Environment response to an action."""

    model_config = {"extra": "forbid"}

    result: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    phraseology_ok: bool = False
    issues: list[str] = Field(default_factory=list)


class AircraftState(BaseModel):
    """Deterministic aircraft state."""

    model_config = {"extra": "forbid"}

    callsign: str = Field(..., min_length=1, max_length=20)
    x_ft: float = Field(..., ge=-10000.0, le=10000.0)
    y_ft: float = Field(..., ge=-10000.0, le=10000.0)
    heading_deg: float = Field(..., ge=0.0, le=360.0)
    altitude_ft: float = Field(..., ge=0.0, le=45000.0)
    speed_kt: float = Field(..., ge=0.0, le=600.0)
    phase: LifecyclePhase
    assigned_runway: str | None = None
    assigned_gate: str | None = None
    wake_category: str = Field(default="M", max_length=1)

    @field_validator("heading_deg")
    @classmethod
    def heading_range(cls, v: float) -> float:
        if not (0.0 <= v < 360.0):
            raise ValueError(f"heading_deg must be 0 <= v < 360, got {v}")
        return v

    @field_validator("altitude_ft")
    @classmethod
    def altitude_range(cls, v: float) -> float:
        if not (0.0 <= v <= 45000.0):
            raise ValueError(f"altitude_ft must be 0 <= v <= 45000, got {v}")
        return v

    @field_validator("speed_kt")
    @classmethod
    def speed_range(cls, v: float) -> float:
        if not (0.0 <= v <= 600.0):
            raise ValueError(f"speed_kt must be 0 <= v <= 600, got {v}")
        return v


class State(BaseModel):
    """Full environment state."""

    model_config = {"extra": "forbid"}

    phase: LifecyclePhase
    aircraft: dict[str, AircraftState] = Field(default_factory=dict)
    episode_id: str = Field(..., min_length=1)
    step_count: int = Field(default=0, ge=0)
    task_id: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)
