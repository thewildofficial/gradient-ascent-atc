"""Departure task: pushback + taxi-out + release sequencing."""

from typing import Any

from pydantic import BaseModel, Field

from src.models import Action, ClearanceType, LifecyclePhase, Observation
from src.rewards import RewardCalculator
from src.state_machine import LifecycleState


# Departure-specific phases (excluding AT_GATE which is the starting point)
DEPARTURE_PHASES = [
    LifecyclePhase.PUSHBACK,
    LifecyclePhase.TAXI_OUT,
    LifecyclePhase.DEPARTURE_QUEUE,
    LifecyclePhase.TAKEOFF,
    LifecyclePhase.DEPARTED,
]

# Number of departure phases for completion bonus
NUM_DEPARTURE_PHASES = 5


class DepartureTask(BaseModel):
    """Departure task covering outbound lifecycle.

    Phase sequence: AT_GATE → PUSHBACK → TAXI_OUT → DEPARTURE_QUEUE → TAKEOFF → DEPARTED

    The aircraft starts at AT_GATE (arrived from prior arrival task) and the agent
    guides it through: pushback clearance → taxi to runway → queue → release → takeoff.

    Episode ends when DEPARTED phase is reached.
    """

    model_config = {"extra": "forbid"}

    task_id: str = Field(default="departure")

    def reset(
        self, seed: int | None = None, episode_id: str | None = None
    ) -> LifecycleState:
        """Reset to initial departure state.

        Args:
            seed: Optional seed for deterministic scenario
            episode_id: Optional episode identifier

        Returns:
            Initial LifecycleState with phase=AT_GATE
        """
        # Import here to avoid circular dependency
        from src.tasks.registry import ScenarioFixtureFactory

        state_dict, _ = ScenarioFixtureFactory.build_departure_fixture(seed=seed or 0)

        # Build LifecycleState from fixture dict
        aircraft_states: dict[str, Any] = {}
        for callsign, ac_data in state_dict.get("aircraft", {}).items():
            from src.models import AircraftState

            aircraft_states[callsign] = AircraftState(**ac_data)

        state = LifecycleState(
            phase=LifecyclePhase(state_dict["phase"]),
            aircraft_states=aircraft_states,
            episode_id=episode_id or state_dict.get("episode_id", ""),
            step_count=0,
            task_id=state_dict.get("task_id", self.task_id),
            completed_phases=[],
            metadata=state_dict.get("metadata", {}),
        )

        # Mark AT_GATE as completed since we're starting there
        state.completed_phases.append(LifecyclePhase.AT_GATE)

        return state

    def step(
        self, state: LifecycleState, action: Action
    ) -> tuple[LifecycleState, Observation]:
        """Execute a departure step.

        Args:
            state: Current lifecycle state
            action: Action to execute

        Returns:
            Tuple of (updated state, observation)
        """
        # Import here to avoid circular dependency
        from src.airport_schema import AirportSchemaLoader

        # Get or create a minimal schema for state machine operations
        try:
            schema = AirportSchemaLoader.load("gatwick")
        except Exception:
            # Fallback to minimal schema if gatwick not available
            from src.airport_schema import AirportSchema, AirportNode, NodeType

            schema = AirportSchema(
                airport_code="DUMMY",
                nodes={
                    "GATE_A1": AirportNode(
                        id="GATE_A1",
                        node_type=NodeType.GATE,
                        x_ft=0.0,
                        y_ft=0.0,
                    ),
                    "DQ_E": AirportNode(
                        id="DQ_E",
                        node_type=NodeType.DEPARTURE_QUEUE,
                        x_ft=1000.0,
                        y_ft=0.0,
                    ),
                },
                edges=[],
                runways=[{"id": "RWY27L", "heading_deg": 270.0}],
                gates=[{"id": "GATE_A1", "x_ft": 0.0, "y_ft": 0.0}],
            )

        # Create a temporary state machine for departure phases only
        sm = _DepartureStateMachine(schema=schema, seed=state.metadata.get("seed"))

        # Set state directly on the state machine
        sm._state = state

        # Execute step
        new_state, obs = sm.step(action)

        return new_state, obs

    def is_terminal(self, state: LifecycleState) -> bool:
        """Check if episode is complete.

        Args:
            state: Current lifecycle state

        Returns:
            True if in DEPARTED phase
        """
        return state.phase == LifecyclePhase.DEPARTED

    def get_legal_actions(self, state: LifecycleState) -> list[Action]:
        """Get legal actions for current departure phase.

        Args:
            state: Current lifecycle state

        Returns:
            List of valid Action objects for the current phase
        """
        from src.airport_schema import AirportSchema, AirportNode, NodeType

        schema = AirportSchema(
            airport_code="DUMMY",
            nodes={
                "GATE_A1": AirportNode(
                    id="GATE_A1",
                    node_type=NodeType.GATE,
                    x_ft=0.0,
                    y_ft=0.0,
                ),
                "DQ_E": AirportNode(
                    id="DQ_E",
                    node_type=NodeType.DEPARTURE_QUEUE,
                    x_ft=1000.0,
                    y_ft=0.0,
                ),
            },
            edges=[],
            runways=[{"id": "RWY27L", "heading_deg": 270.0}],
            gates=[{"id": "GATE_A1", "x_ft": 0.0, "y_ft": 0.0}],
        )

        sm = _DepartureStateMachine(schema=schema, seed=state.metadata.get("seed"))
        sm._state = state

        return sm.get_legal_actions(state)


class DepartureGrader:
    """Grader for departure task.

    Evaluates departure episodes using:
    - RewardCalculator from src.rewards
    - Safety blocks for unsafe operations (apron conflict, unsafe runway release)
    - Completion bonus for finishing all 5 departure phases
    """

    COMPLETION_BONUS = 0.1

    def __init__(self) -> None:
        self._reward_calculator = RewardCalculator()

    def grade(self, state: LifecycleState, rewards: list[float]) -> float:
        """Compute final departure score.

        Args:
            state: Final lifecycle state
            rewards: List of per-step rewards from RewardCalculator

        Returns:
            Final score in [0.0, 1.0]
        """
        # Safety block: unsafe apron conflict
        if self._has_apron_conflict(state):
            return 0.0

        # Safety block: unsafe runway release
        if self._has_unsafe_runway_release(state):
            return 0.0

        # Compute base score from rewards
        if not rewards:
            base_score = 0.0
        else:
            total = sum(rewards)
            base_score = total / len(rewards)

        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, base_score))

        # Completion bonus: all 5 departure phases completed
        if self._all_departure_phases_completed(state):
            score = min(1.0, score + self.COMPLETION_BONUS)

        return score

    def grade_step(
        self, state: LifecycleState, action: Action, obs: Observation
    ) -> float:
        """Grade a single step and return reward.

        Args:
            state: Current lifecycle state
            action: Action taken
            obs: Observation returned

        Returns:
            Step reward in [0.0, 1.0]
        """
        _, reward = self._reward_calculator.compute_reward(state, action, obs)
        return reward

    def _has_apron_conflict(self, state: LifecycleState) -> bool:
        """Check for unsafe apron conflict.

        Args:
            state: Current lifecycle state

        Returns:
            True if there's an unsafe apron conflict
        """
        # Check metadata for apron conflict indicator
        if state.metadata.get("apron_conflict"):
            return True

        # Check for collision issues in metadata
        if state.metadata.get("collision"):
            return True

        return False

    def _has_unsafe_runway_release(self, state: LifecycleState) -> bool:
        """Check for unsafe runway release.

        Args:
            state: Current lifecycle state

        Returns:
            True if runway release was unsafe
        """
        # Check metadata for unsafe runway release indicator
        if state.metadata.get("unsafe_runway_release"):
            return True

        # Check for runway incursion indicator
        if state.metadata.get("runway_incursion"):
            return True

        # If we're at TAKEOFF or beyond without proper sequencing, flag it
        if state.phase in (LifecyclePhase.TAKEOFF, LifecyclePhase.DEPARTED):
            line_up = state.metadata.get("line_up_confirmed", False)
            takeoff = state.metadata.get("takeoff_confirmed", False)
            if not line_up or not takeoff:
                return True

        return False

    def _all_departure_phases_completed(self, state: LifecycleState) -> bool:
        """Check if all 5 departure phases were completed.

        The 5 phases are: PUSHBACK, TAXI_OUT, DEPARTURE_QUEUE, TAKEOFF, DEPARTED

        Args:
            state: Final lifecycle state

        Returns:
            True if all phases completed
        """
        required = {
            LifecyclePhase.PUSHBACK,
            LifecyclePhase.TAXI_OUT,
            LifecyclePhase.DEPARTURE_QUEUE,
            LifecyclePhase.TAKEOFF,
            LifecyclePhase.DEPARTED,
        }

        completed = set(state.completed_phases)
        return required.issubset(completed)


class _DepartureStateMachine:
    """Lightweight state machine for departure-only phases.

    Wraps the core state machine logic focused on departure lifecycle.
    """

    def __init__(self, schema: Any, seed: int | None = None) -> None:
        """Initialize departure state machine.

        Args:
            schema: AirportSchema for topology
            seed: Optional random seed
        """
        import random

        self._schema = schema
        self._seed = seed
        self._rng = random.Random(seed)
        self._state: LifecycleState | None = None
        self._route_index: int = 0
        self._pushback_complete: bool = False

    def step(self, action: Action) -> tuple[LifecycleState, Observation]:
        """Execute a departure action.

        Args:
            action: Controller action

        Returns:
            Tuple of (updated state, observation)
        """
        if self._state is None:
            raise RuntimeError("State machine not initialized")

        self._state.step_count += 1
        current_phase = self._state.phase

        # Terminal state
        if current_phase == LifecyclePhase.DEPARTED:
            return self._state, Observation(
                result="episode_complete",
                score=1.0,
                phraseology_ok=True,
                issues=[],
            )

        reward = 0.0
        issues: list[str] = []

        # Departure phase handling
        if current_phase == LifecyclePhase.AT_GATE:
            reward, issues = self._step_at_gate(action)
        elif current_phase == LifecyclePhase.PUSHBACK:
            reward, issues = self._step_pushback(action)
        elif current_phase == LifecyclePhase.TAXI_OUT:
            reward, issues = self._step_taxi_out(action)
        elif current_phase == LifecyclePhase.DEPARTURE_QUEUE:
            reward, issues = self._step_departure_queue(action)
        elif current_phase == LifecyclePhase.TAKEOFF:
            reward, issues = self._step_takeoff(action)

        if issues:
            return self._state, Observation(
                result="illegal_transition",
                score=0.0,
                phraseology_ok=False,
                issues=issues,
            )

        return self._state, Observation(
            result=f"phase_{current_phase.value}_continued",
            score=reward,
            phraseology_ok=True,
            issues=[],
        )

    def get_legal_actions(self, state: LifecycleState) -> list[Action]:
        """Get legal actions for current phase.

        Args:
            state: Current lifecycle state

        Returns:
            List of valid actions
        """
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []

        phase = state.phase

        if phase == LifecyclePhase.AT_GATE:
            return [
                Action(
                    clearance_type=ClearanceType.PUSHBACK,
                    target_callsign=aircraft.callsign,
                    pushback_direction="back",
                )
            ]
        elif phase == LifecyclePhase.PUSHBACK:
            direction = state.metadata.get("pushback_direction", "back")
            return [
                Action(
                    clearance_type=ClearanceType.PUSHBACK,
                    target_callsign=aircraft.callsign,
                    pushback_direction=direction,
                )
            ]
        elif phase == LifecyclePhase.TAXI_OUT:
            return [
                Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign=aircraft.callsign,
                    route=["DQ_E", "RE_E"],
                )
            ]
        elif phase == LifecyclePhase.DEPARTURE_QUEUE:
            actions = []
            if not state.metadata.get("line_up_confirmed", False):
                actions.append(
                    Action(
                        clearance_type=ClearanceType.LINE_UP,
                        target_callsign=aircraft.callsign,
                    )
                )
            if state.metadata.get(
                "line_up_confirmed", False
            ) and not state.metadata.get("takeoff_confirmed", False):
                actions.append(
                    Action(
                        clearance_type=ClearanceType.TAKEOFF,
                        target_callsign=aircraft.callsign,
                    )
                )
            return actions
        elif phase == LifecyclePhase.TAKEOFF:
            return [
                Action(
                    clearance_type=ClearanceType.TAKEOFF,
                    target_callsign=aircraft.callsign,
                )
            ]

        return []

    def _step_at_gate(self, action: Action) -> tuple[float, list[str]]:
        """Handle AT_GATE phase."""
        assert self._state is not None
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        if action.clearance_type == ClearanceType.PUSHBACK:
            if action.pushback_direction:
                self._state.phase = LifecyclePhase.PUSHBACK
                self._state.completed_phases.append(LifecyclePhase.AT_GATE)
                self._state.metadata["pushback_direction"] = action.pushback_direction
                aircraft.phase = LifecyclePhase.PUSHBACK
                aircraft.speed_kt = 5.0
                return 0.1, []

        return 0.0, []

    def _step_pushback(self, action: Action) -> tuple[float, list[str]]:
        """Handle PUSHBACK phase."""
        assert self._state is not None
        if not action.pushback_direction:
            return 0.0, ["missing_pushback_direction"]

        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Move in pushback direction
        direction_sign = -1.0 if action.pushback_direction in ("back", "north") else 1.0
        aircraft.y_ft += direction_sign * 10.0

        # Check if pushback complete
        if abs(aircraft.y_ft) > 400.0:
            self._pushback_complete = True
            self._state.phase = LifecyclePhase.TAXI_OUT
            self._state.completed_phases.append(LifecyclePhase.PUSHBACK)
            aircraft.phase = LifecyclePhase.TAXI_OUT
            aircraft.speed_kt = 20.0
            return 0.1, []

        return 0.0, []

    def _step_taxi_out(self, action: Action) -> tuple[float, list[str]]:
        """Handle TAXI_OUT phase."""
        assert self._state is not None
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        if action.clearance_type == ClearanceType.TAXI and action.route:
            if not self._state.metadata.get("taxi_route"):
                self._state.metadata["taxi_route"] = action.route
                self._route_index = 0

            # Move along route
            if self._route_index < len(action.route):
                target_id = action.route[self._route_index]
                target_node = self._schema.nodes.get(target_id)

                if target_node:
                    import math

                    dx = target_node.x_ft - aircraft.x_ft
                    dy = target_node.y_ft - aircraft.y_ft
                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist < 100.0:
                        self._route_index += 1
                        if self._route_index >= len(action.route):
                            # Reached departure queue
                            self._state.phase = LifecyclePhase.DEPARTURE_QUEUE
                            self._state.completed_phases.append(LifecyclePhase.TAXI_OUT)
                            aircraft.phase = LifecyclePhase.DEPARTURE_QUEUE
                            aircraft.speed_kt = 0.0
                            return 0.1, []
                    else:
                        # Move towards target
                        target_heading = math.degrees(math.atan2(dx, dy))
                        heading_rad = math.radians(target_heading)
                        aircraft.heading_deg = target_heading
                        distance_ft = 20.0 * (6076.1 / 3600)
                        aircraft.x_ft += distance_ft * math.sin(heading_rad)
                        aircraft.y_ft += distance_ft * math.cos(heading_rad)

        return 0.0, []

    def _step_departure_queue(self, action: Action) -> tuple[float, list[str]]:
        """Handle DEPARTURE_QUEUE phase."""
        assert self._state is not None
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        if action.clearance_type == ClearanceType.LINE_UP:
            self._state.metadata["line_up_confirmed"] = True
        elif action.clearance_type == ClearanceType.TAKEOFF:
            self._state.metadata["takeoff_confirmed"] = True

        if self._state.metadata.get(
            "line_up_confirmed", False
        ) and self._state.metadata.get("takeoff_confirmed", False):
            self._state.phase = LifecyclePhase.TAKEOFF
            self._state.completed_phases.append(LifecyclePhase.DEPARTURE_QUEUE)
            aircraft.phase = LifecyclePhase.TAKEOFF
            aircraft.speed_kt = 30.0
            return 0.1, []

        return 0.0, []

    def _step_takeoff(self, action: Action) -> tuple[float, list[str]]:
        """Handle TAKEOFF phase."""
        assert self._state is not None
        if not self._state.metadata.get(
            "line_up_confirmed", False
        ) or not self._state.metadata.get("takeoff_confirmed", False):
            return 0.0, ["missing_takeoff_clearance"]

        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Accelerate and takeoff
        aircraft.speed_kt = min(aircraft.speed_kt + 20.0, 160.0)
        aircraft.y_ft -= aircraft.speed_kt * 0.5

        # Check if departed (past runway end)
        runway_length = 9850.0
        if aircraft.y_ft < -runway_length:
            self._state.phase = LifecyclePhase.DEPARTED
            self._state.completed_phases.append(LifecyclePhase.TAKEOFF)
            self._state.completed_phases.append(LifecyclePhase.DEPARTED)
            aircraft.phase = LifecyclePhase.DEPARTED
            aircraft.speed_kt = 0.0
            return 0.1, []

        return 0.0, []

    def _get_primary_aircraft(self):
        """Get primary aircraft from state."""
        if self._state and self._state.aircraft_states:
            return next(iter(self._state.aircraft_states.values()))
        return None
