"""Task registry and scenario fixtures."""

import random
from typing import Literal

from pydantic import BaseModel, Field

from src.models import Action, AircraftState, ClearanceType, LifecyclePhase, State


class TaskInfo(BaseModel):
    """Task information metadata.

    Attributes:
        task_id: Unique identifier for the task.
        name: Human-readable task name.
        description: Description of what the task entails.
        difficulty: Difficulty level (easy, medium, hard).
        initial_state_fn: Name of the factory function for this task's initial state.
    """

    model_config = {"extra": "forbid"}

    task_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    initial_state_fn: str = Field(..., min_length=1)


class TaskRegistry:
    """Registry for managing available tasks.

    Provides task registration, retrieval, and enumeration capabilities.
    Ships with 3 pre-registered tasks: departure (easy), arrival (hard), integrated (medium).
    """

    REGISTRY: dict[str, TaskInfo] = {}

    @classmethod
    def register(cls, task: TaskInfo) -> None:
        """Register a task in the registry.

        Args:
            task: TaskInfo instance to register.
        """
        cls.REGISTRY[task.task_id] = task

    @classmethod
    def get(cls, task_id: str) -> TaskInfo:
        """Retrieve a task by its ID.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            TaskInfo for the requested task.

        Raises:
            KeyError: If task_id is not found in the registry.
        """
        if task_id not in cls.REGISTRY:
            raise KeyError(f"Task '{task_id}' not found in registry")
        return cls.REGISTRY[task_id]

    @classmethod
    def list_tasks(cls) -> list[TaskInfo]:
        """Return all registered tasks.

        Returns:
            List of all TaskInfo instances in the registry.
        """
        return list(cls.REGISTRY.values())

    @classmethod
    def list_by_difficulty(cls, difficulty: str) -> list[TaskInfo]:
        """Return tasks filtered by difficulty level.

        Args:
            difficulty: Difficulty level to filter by (easy, medium, hard).

        Returns:
            List of TaskInfo instances matching the specified difficulty.
        """
        return [t for t in cls.REGISTRY.values() if t.difficulty == difficulty]


# Pre-register the three canonical tasks
TaskRegistry.register(
    TaskInfo(
        task_id="departure",
        name="Departure Task",
        description="pushback → taxi-out → departure queue → release sequencing",
        difficulty="easy",
        initial_state_fn="build_departure_fixture",
    )
)

TaskRegistry.register(
    TaskInfo(
        task_id="arrival",
        name="Arrival Task",
        description="landing → handoff → taxi-in → docking",
        difficulty="hard",
        initial_state_fn="build_arrival_fixture",
    )
)

TaskRegistry.register(
    TaskInfo(
        task_id="integrated",
        name="Integrated Task",
        description="full arrival + turnaround + departure",
        difficulty="medium",
        initial_state_fn="build_integrated_fixture",
    )
)

TaskRegistry.register(
    TaskInfo(
        task_id="peak_traffic",
        name="Peak Traffic",
        description="3 aircraft simultaneously — prevent collisions, coordinate all to completion",
        difficulty="hard",
        initial_state_fn="build_peak_traffic_fixture",
    )
)


class ScenarioFixtureFactory:
    """Factory for creating deterministic scenario fixtures.

    All fixtures are deterministic: same seed produces identical output.
    Uses random.Random(seed) for reproducible scenario generation.
    """

    @staticmethod
    def _seeded_random(seed: int) -> random.Random:
        """Create a seeded random generator.

        Args:
            seed: Seed value for reproducibility.

        Returns:
            random.Random instance seeded with the given value.
        """
        return random.Random(seed)

    @classmethod
    def build(cls, task_id: str, seed: int) -> tuple[dict, list[dict]]:
        """Build a scenario fixture for a given task.

        Args:
            task_id: The task identifier.
            seed: Seed for deterministic scenario generation.

        Returns:
            Tuple of (initial_state, expected_action_sequence).

        Raises:
            KeyError: If task_id is not found in the registry.
        """
        task = TaskRegistry.get(task_id)
        fn_name = task.initial_state_fn

        if fn_name == "build_departure_fixture":
            return cls.build_departure_fixture(seed)
        elif fn_name == "build_arrival_fixture":
            return cls.build_arrival_fixture(seed)
        elif fn_name == "build_integrated_fixture":
            return cls.build_integrated_fixture(seed)
        elif fn_name == "build_peak_traffic_fixture":
            return cls.build_peak_traffic_fixture(seed)
        else:
            raise ValueError(f"Unknown initial_state_fn: {fn_name}")

    @classmethod
    def build_departure_fixture(cls, seed: int) -> tuple[dict, list[dict]]:
        """Build a departure scenario fixture.

        Scenario phases: pushback → taxi-out → departure queue → release sequencing

        Args:
            seed: Seed for deterministic scenario generation.

        Returns:
            Tuple of (initial_state_dict, action_sequence_list).
        """
        rng = cls._seeded_random(seed)

        # Generate callsign
        airline_prefixes = ["BAW", "EZY", "AFR", "DLH", "UAE"]
        suffix = rng.randint(100, 999)
        callsign = f"{rng.choice(airline_prefixes)}{suffix}"

        # Initial aircraft state at gate
        initial_state = {
            "phase": LifecyclePhase.AT_GATE,
            "aircraft": {
                callsign: {
                    "callsign": callsign,
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": rng.uniform(0.0, 360.0),
                    "altitude_ft": 0.0,
                    "speed_kt": 0.0,
                    "phase": LifecyclePhase.AT_GATE,
                    "assigned_runway": "RWY27L",
                    "assigned_gate": "GATE_A1",
                    "wake_category": rng.choice(["M", "H", "L"]),
                }
            },
            "episode_id": f"dep-{seed}-{callsign}",
            "step_count": 0,
            "task_id": "departure",
            "metadata": {"seed": seed, "scenario": "departure"},
        }

        # Expected action sequence for departure scenario
        actions = [
            {
                "clearance_type": ClearanceType.PUSHBACK,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": "north",
                "hold_short": False,
                "runway": None,
            },
            {
                "clearance_type": ClearanceType.TAXI,
                "target_callsign": callsign,
                "route": ["TAXIWAY_A", "TAXIWAY_B", "DEPARTURE_QUEUE"],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.HOLD_SHORT,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": True,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.LINE_UP,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.TAKEOFF,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
        ]

        return initial_state, actions

    @classmethod
    def build_arrival_fixture(cls, seed: int) -> tuple[dict, list[dict]]:
        """Build an arrival scenario fixture.

        Scenario phases: landing → handoff → taxi-in → docking

        Args:
            seed: Seed for deterministic scenario generation.

        Returns:
            Tuple of (initial_state_dict, action_sequence_list).
        """
        rng = cls._seeded_random(seed)

        # Generate callsign
        airline_prefixes = ["BAW", "EZY", "AFR", "DLH", "UAE"]
        suffix = rng.randint(100, 999)
        callsign = f"{rng.choice(airline_prefixes)}{suffix}"

        import math

        altitude_ft = 5000.0
        distance_to_runway = altitude_ft / math.tan(math.radians(3.0))

        initial_state = {
            "phase": LifecyclePhase.APPROACH,
            "aircraft": {
                callsign: {
                    "callsign": callsign,
                    "x_ft": rng.uniform(-500.0, 500.0),
                    "y_ft": distance_to_runway + rng.uniform(-500.0, 500.0),
                    "heading_deg": 270.0,
                    "altitude_ft": altitude_ft,
                    "speed_kt": 140.0,
                    "phase": LifecyclePhase.APPROACH,
                    "assigned_runway": "RWY27L",
                    "assigned_gate": "GATE_B2",
                    "wake_category": rng.choice(["M", "H", "L"]),
                }
            },
            "episode_id": f"arr-{seed}-{callsign}",
            "step_count": 0,
            "task_id": "arrival",
            "metadata": {"seed": seed, "scenario": "arrival"},
        }

        # Expected action sequence for arrival scenario
        actions = [
            {
                "clearance_type": ClearanceType.LANDING,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.TAXI,
                "target_callsign": callsign,
                "route": ["RUNWAY_EXIT", "TAXIWAY_C", "TAXIWAY_D", "GATE_B2"],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": None,
            },
            {
                "clearance_type": ClearanceType.TAKEOFF,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
        ]

        return initial_state, actions

    @classmethod
    def build_peak_traffic_fixture(cls, seed: int) -> tuple[dict, list[dict]]:
        """Build a peak traffic scenario fixture with 3 aircraft.

        Args:
            seed: Seed for deterministic scenario generation.

        Returns:
            Tuple of (initial_state_dict, action_sequence_list).
        """
        rng = cls._seeded_random(seed)

        airline_prefixes = ["BAW", "EZY", "AFR", "DLH", "UAE"]
        callsigns = [
            f"{rng.choice(airline_prefixes)}{rng.randint(100, 999)}" for _ in range(3)
        ]

        import math

        altitude_ft = 5000.0
        distance_to_runway = altitude_ft / math.tan(math.radians(3.0))

        initial_state = {
            "phase": LifecyclePhase.APPROACH,
            "aircraft": {
                callsigns[0]: {
                    "callsign": callsigns[0],
                    "x_ft": rng.uniform(-500.0, 500.0),
                    "y_ft": distance_to_runway + rng.uniform(-500.0, 500.0),
                    "heading_deg": 270.0,
                    "altitude_ft": altitude_ft,
                    "speed_kt": 140.0,
                    "phase": LifecyclePhase.APPROACH,
                    "assigned_runway": "RWY27L",
                    "assigned_gate": "GATE_B2",
                    "wake_category": rng.choice(["M", "H", "L"]),
                },
                callsigns[1]: {
                    "callsign": callsigns[1],
                    "x_ft": rng.uniform(-100.0, 100.0),
                    "y_ft": rng.uniform(-100.0, 100.0),
                    "heading_deg": 270.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 20.0,
                    "phase": LifecyclePhase.TAXI_IN,
                    "assigned_runway": "RWY27L",
                    "assigned_gate": "GATE_A1",
                    "wake_category": rng.choice(["M", "H", "L"]),
                },
                callsigns[2]: {
                    "callsign": callsigns[2],
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 270.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 0.0,
                    "phase": LifecyclePhase.AT_GATE,
                    "assigned_runway": "RWY27L",
                    "assigned_gate": "GATE_C3",
                    "wake_category": rng.choice(["M", "H", "L"]),
                },
            },
            "episode_id": f"peak-{seed}",
            "step_count": 0,
            "task_id": "peak_traffic",
            "metadata": {"seed": seed, "scenario": "peak_traffic"},
        }

        actions = [
            {
                "clearance_type": ClearanceType.LANDING,
                "target_callsign": callsigns[0],
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
        ]

        return initial_state, actions

    @classmethod
    def build_integrated_fixture(cls, seed: int) -> tuple[dict, list[dict]]:
        """Build an integrated full-lifecycle scenario fixture.

        Scenario phases: full arrival + turnaround + departure

        Args:
            seed: Seed for deterministic scenario generation.

        Returns:
            Tuple of (initial_state_dict, action_sequence_list).
        """
        rng = cls._seeded_random(seed)

        # Generate callsign
        airline_prefixes = ["BAW", "EZY", "AFR", "DLH", "UAE"]
        suffix = rng.randint(100, 999)
        callsign = f"{rng.choice(airline_prefixes)}{suffix}"

        import math

        altitude_ft = 5000.0
        distance_to_runway = altitude_ft / math.tan(math.radians(3.0))

        initial_state = {
            "phase": LifecyclePhase.APPROACH,
            "aircraft": {
                callsign: {
                    "callsign": callsign,
                    "x_ft": rng.uniform(-500.0, 500.0),
                    "y_ft": distance_to_runway + rng.uniform(-500.0, 500.0),
                    "heading_deg": 270.0,
                    "altitude_ft": altitude_ft,
                    "speed_kt": 140.0,
                    "phase": LifecyclePhase.APPROACH,
                    "assigned_runway": "RWY27L",
                    "assigned_gate": "GATE_C3",
                    "wake_category": rng.choice(["M", "H", "L"]),
                }
            },
            "episode_id": f"int-{seed}-{callsign}",
            "step_count": 0,
            "task_id": "integrated",
            "metadata": {"seed": seed, "scenario": "integrated"},
        }

        # Expected action sequence for integrated scenario (arrival + departure)
        actions = [
            # Arrival phase
            {
                "clearance_type": ClearanceType.LANDING,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.TAXI,
                "target_callsign": callsign,
                "route": ["RUNWAY_EXIT", "TAXIWAY_C", "TAXIWAY_D", "GATE_C3"],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": None,
            },
            {
                "clearance_type": ClearanceType.HOLD_SHORT,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": True,
                "runway": None,
            },
            # Turnaround complete, now departure phase
            {
                "clearance_type": ClearanceType.PUSHBACK,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": "south",
                "hold_short": False,
                "runway": None,
            },
            {
                "clearance_type": ClearanceType.TAXI,
                "target_callsign": callsign,
                "route": ["TAXIWAY_D", "TAXIWAY_B", "DEPARTURE_QUEUE"],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.HOLD_SHORT,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": True,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.LINE_UP,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
            {
                "clearance_type": ClearanceType.TAKEOFF,
                "target_callsign": callsign,
                "route": [],
                "readback_required": True,
                "pushback_direction": None,
                "hold_short": False,
                "runway": "RWY27L",
            },
        ]

        return initial_state, actions
