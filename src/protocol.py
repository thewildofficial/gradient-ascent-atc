"""Normalized hybrid ATC protocol definitions."""

from typing import ClassVar

from pydantic import BaseModel, Field

from src.models import Action, ClearanceType


class ClearanceDefinition(BaseModel):
    type: ClearanceType
    required_fields: list[str] = Field(default_factory=list)
    phraseology_template: str = ""
    readback_required: bool = False


class ProtocolValidator(BaseModel):
    _required_fields: dict[ClearanceType, list[str]] = {
        ClearanceType.PUSHBACK: ["pushback_direction"],
        ClearanceType.TAXI: ["route"],
        ClearanceType.HOLD_SHORT: ["runway"],
        ClearanceType.CROSS_RUNWAY: ["runway"],
        ClearanceType.LINE_UP: ["runway"],
        ClearanceType.TAKEOFF: ["runway"],
        ClearanceType.LANDING: ["runway"],
    }

    _route_valid_edges: dict[tuple[str, str], set[str]] = {
        ("taxiway", "taxiway"): {"taxi"},
        ("taxiway", "runway"): {"cross", "enter", "line_up"},
        ("runway", "taxiway"): {"exit", "cross"},
        ("runway", "runway"): {"takeoff", "landing", "line_up"},
        ("gate", "taxiway"): {"taxi", "pushback"},
        ("taxiway", "gate"): {"taxi"},
    }

    def validate_clearance(self, clearance: Action) -> tuple[bool, str | None]:
        required = self._required_fields.get(clearance.clearance_type, [])
        for field in required:
            value = getattr(clearance, field, None)
            if field == "route":
                if not value or len(value) == 0:
                    return (
                        False,
                        f"Clearance {clearance.clearance_type.value} requires non-empty route",
                    )
            elif value is None:
                return (
                    False,
                    f"Clearance {clearance.clearance_type.value} requires {field}",
                )
        return True, None

    def is_valid_route_segment(
        self, node1_type: str, node2_type: str, edge_type: str
    ) -> bool:
        key = (node1_type, node2_type)
        if key not in self._route_valid_edges:
            return False
        return edge_type in self._route_valid_edges[key]


class HandoffProtocol(BaseModel):
    FREQUENCIES: ClassVar[dict[str, float]] = {
        "ground": 121.9,
        "tower": 118.9,
        "approach": 119.9,
        "departure": 132.6,
    }

    _valid_handoffs: dict[tuple[float, float], set[str]] = {
        (121.9, 118.9): {"taxi_out", "pushback", "departure_queue"},
        (118.9, 132.6): {"takeoff", "departed"},
        (119.9, 118.9): {"landing", "approach"},
        (118.9, 119.9): {"departure_queue"},
    }

    def validate_handoff(
        self, from_freq: float, to_freq: float, current_phase: str
    ) -> bool:
        key = (from_freq, to_freq)
        if key not in self._valid_handoffs:
            return False
        return current_phase in self._valid_handoffs[key]
