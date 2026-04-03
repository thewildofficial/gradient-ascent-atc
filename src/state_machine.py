"""Full-lifecycle state machine for aircraft operations."""

import random
from typing import Any

from src.airport_schema import AirportSchema
from src.models import (
    Action,
    AircraftState,
    ClearanceType,
    LifecyclePhase,
    Observation,
)


# Phase ordering for the full lifecycle
_PHASE_SEQUENCE = [
    LifecyclePhase.APPROACH,
    LifecyclePhase.LANDING,
    LifecyclePhase.ARRIVAL_HANDOFF,
    LifecyclePhase.TAXI_IN,
    LifecyclePhase.DOCKING,
    LifecyclePhase.AT_GATE,
    LifecyclePhase.PUSHBACK,
    LifecyclePhase.TAXI_OUT,
    LifecyclePhase.DEPARTURE_QUEUE,
    LifecyclePhase.TAKEOFF,
    LifecyclePhase.DEPARTED,
]

# Landing threshold altitude in feet
LANDING_ALTITUDE_THRESHOLD_FT = 50.0

# Docking duration in seconds
DOCKING_DURATION_S = 30.0

# Turnaround delay at gate in seconds
TURNAROUND_DELAY_S = 60.0

# Default simulation timestep in seconds
DEFAULT_DT_S = 1.0


class LifecycleState:
    """Full lifecycle state for the ATC environment.

    Attributes:
        phase: Current lifecycle phase
        aircraft_states: Dict of callsign -> AircraftState
        episode_id: Unique episode identifier
        step_count: Number of steps taken in this episode
        task_id: Task being performed
        completed_phases: List of phases successfully completed
        metadata: Additional phase-specific data
    """

    __slots__ = (
        "phase",
        "aircraft_states",
        "episode_id",
        "step_count",
        "task_id",
        "completed_phases",
        "metadata",
    )

    def __init__(
        self,
        phase: LifecyclePhase = LifecyclePhase.APPROACH,
        aircraft_states: dict[str, AircraftState] | None = None,
        episode_id: str = "",
        step_count: int = 0,
        task_id: str = "",
        completed_phases: list[LifecyclePhase] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.aircraft_states = aircraft_states or {}
        self.episode_id = episode_id
        self.step_count = step_count
        self.task_id = task_id
        self.completed_phases = completed_phases or []
        self.metadata = metadata or {}


class FullLifecycleStateMachine:
    """State machine driving the full operational lifecycle.

    Implements the 11-phase lifecycle:
    APPROACH → LANDING → ARRIVAL_HANDOFF → TAXI_IN → DOCKING →
    AT_GATE → PUSHBACK → TAXI_OUT → DEPARTURE_QUEUE → TAKEOFF → DEPARTED

    Each phase has enter() and exit() methods that manage state transitions
    and aircraft physics updates.
    """

    def __init__(self, schema: AirportSchema, seed: int | None = None) -> None:
        """Initialize the state machine.

        Args:
            schema: Airport topology schema
            seed: Optional random seed for deterministic behavior
        """
        self._schema = schema
        self._seed = seed
        self._rng = random.Random(seed)
        self._state = LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={},
            episode_id="",
            step_count=0,
            task_id="",
        )
        self._docking_timer: float = 0.0
        self._turnaround_timer: float = 0.0
        self._current_node: str | None = None
        self._route_index: int = 0
        self._pushback_complete: bool = False

    def reset(self, task_id: str, episode_id: str) -> LifecycleState:
        """Reset to initial state for the given task.

        Args:
            task_id: Task identifier
            episode_id: Episode identifier

        Returns:
            Initial LifecycleState
        """
        self._rng = random.Random(self._seed)
        self._docking_timer = 0.0
        self._turnaround_timer = 0.0
        self._route_index = 0
        self._pushback_complete = False

        # Find approach fix and runway threshold from schema
        approach_node = self._find_first_node_by_type("approach_fix")
        threshold_node = self._find_first_node_by_type("landing_threshold")
        gate_node = self._find_first_node_by_type("gate")
        stand_node = self._find_first_node_by_type("stand")

        # Initialize aircraft at approach position
        # Clamp to AircraftState validation limits (-10000 to 10000)
        if approach_node:
            x = max(-10000.0, min(10000.0, approach_node.x_ft))
            y = max(-10000.0, min(10000.0, approach_node.y_ft))
        else:
            x, y = 0.0, 5000.0

        # Get runway heading from first runway
        runway_heading = 270.0
        if self._schema.runways:
            runway_heading = self._schema.runways[0].get("heading_deg", 270.0)

        aircraft = AircraftState(
            callsign="BAW123",
            x_ft=x,
            y_ft=y,
            heading_deg=runway_heading,
            altitude_ft=5000.0,
            speed_kt=250.0,
            phase=LifecyclePhase.APPROACH,
            assigned_runway=self._schema.runways[0]["id"]
            if self._schema.runways
            else "27L",
            assigned_gate=gate_node.id if gate_node else "GATE_A1",
        )

        self._state = LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={aircraft.callsign: aircraft},
            episode_id=episode_id,
            step_count=0,
            task_id=task_id,
            completed_phases=[],
            metadata={
                "approach_node": approach_node.id if approach_node else "APP_FIX_27L",
                "threshold_node": threshold_node.id if threshold_node else "THR_27L",
                "gate_node": gate_node.id if gate_node else "GATE_A1",
                "stand_node": stand_node.id if stand_node else "STAND_101",
                "runway_heading": runway_heading,
                "ground_frequency_confirmed": False,
                "pushback_direction": None,
                "taxi_route": [],
                "line_up_confirmed": False,
                "takeoff_confirmed": False,
            },
        )
        self._current_node = approach_node.id if approach_node else "APP_FIX_27L"
        self._enter_phase(LifecyclePhase.APPROACH)
        return self._state

    def step(self, action: Action) -> tuple[LifecycleState, Observation]:
        """Execute an action and return new state + observation.

        Args:
            action: Controller action to execute

        Returns:
            Tuple of (new LifecycleState, Observation with result/score/issues)
        """
        if self._state is None:
            raise RuntimeError("State machine not initialized. Call reset() first.")
        assert self._state is not None

        self._state.step_count += 1
        current_phase = self._state.phase

        # Terminal state - no further actions needed
        if current_phase == LifecyclePhase.DEPARTED:
            return self._state, Observation(
                result="episode_complete",
                score=1.0,
                phraseology_ok=True,
                issues=[],
            )

        # Phases that don't require specific actions (automated)
        automated_phases = {LifecyclePhase.DOCKING}

        # Validate action is legal for current phase
        legal_actions = self.get_legal_actions(self._state)

        # APPROACH is special: during descent, only LANDING is potentially legal
        # Other clearances (TAXI, etc.) are definitely wrong and should be rejected
        skip_validation = False
        if current_phase == LifecyclePhase.APPROACH and not legal_actions:
            if action.clearance_type == ClearanceType.LANDING:
                skip_validation = True  # Allow LANDING even before altitude threshold
            else:
                return self._state, Observation(
                    result="illegal_transition",
                    score=0.0,
                    phraseology_ok=False,
                    issues=["illegal_transition"],
                )

        if (
            current_phase not in automated_phases
            and action not in legal_actions
            and not skip_validation
        ):
            return self._state, Observation(
                result="illegal_transition",
                score=0.0,
                phraseology_ok=False,
                issues=["illegal_transition"],
            )

        # Execute phase-specific logic
        reward = 0.0
        issues: list[str] = []

        if current_phase == LifecyclePhase.APPROACH:
            reward, issues = self._step_approach(action)
        elif current_phase == LifecyclePhase.LANDING:
            reward, issues = self._step_landing(action)
        elif current_phase == LifecyclePhase.ARRIVAL_HANDOFF:
            reward, issues = self._step_arrival_handoff(action)
        elif current_phase == LifecyclePhase.TAXI_IN:
            reward, issues = self._step_taxi_in(action)
        elif current_phase == LifecyclePhase.DOCKING:
            reward, issues = self._step_docking(action)
        elif current_phase == LifecyclePhase.AT_GATE:
            reward, issues = self._step_at_gate(action)
        elif current_phase == LifecyclePhase.PUSHBACK:
            reward, issues = self._step_pushback(action)
        elif current_phase == LifecyclePhase.TAXI_OUT:
            reward, issues = self._step_taxi_out(action)
        elif current_phase == LifecyclePhase.DEPARTURE_QUEUE:
            reward, issues = self._step_departure_queue(action)
        elif current_phase == LifecyclePhase.TAKEOFF:
            reward, issues = self._step_takeoff(action)
        elif current_phase == LifecyclePhase.DEPARTED:
            return self._state, Observation(
                result="episode_complete",
                score=1.0,
                phraseology_ok=True,
                issues=[],
            )

        # Check for illegal transitions
        if "illegal_transition" in issues:
            return self._state, Observation(
                result="illegal_transition",
                score=0.0,
                phraseology_ok=False,
                issues=issues,
            )

        # Update aircraft physics for applicable phases
        self._update_aircraft_physics(DEFAULT_DT_S)

        return self._state, Observation(
            result=f"phase_{current_phase.value}_continued",
            score=reward,
            phraseology_ok=True,
            issues=issues,
        )

    def get_legal_actions(self, state: LifecycleState) -> list[Action]:
        """Return list of valid actions for the current state.

        Args:
            state: Current lifecycle state

        Returns:
            List of valid Action objects
        """
        if state.phase == LifecyclePhase.APPROACH:
            return self._legal_approach_actions(state)
        elif state.phase == LifecyclePhase.LANDING:
            return self._legal_landing_actions(state)
        elif state.phase == LifecyclePhase.ARRIVAL_HANDOFF:
            return self._legal_arrival_handoff_actions(state)
        elif state.phase == LifecyclePhase.TAXI_IN:
            return self._legal_taxi_in_actions(state)
        elif state.phase == LifecyclePhase.DOCKING:
            return self._legal_docking_actions(state)
        elif state.phase == LifecyclePhase.AT_GATE:
            return self._legal_at_gate_actions(state)
        elif state.phase == LifecyclePhase.PUSHBACK:
            return self._legal_pushback_actions(state)
        elif state.phase == LifecyclePhase.TAXI_OUT:
            return self._legal_taxi_out_actions(state)
        elif state.phase == LifecyclePhase.DEPARTURE_QUEUE:
            return self._legal_departure_queue_actions(state)
        elif state.phase == LifecyclePhase.TAKEOFF:
            return self._legal_takeoff_actions(state)
        elif state.phase == LifecyclePhase.DEPARTED:
            return []
        return []

    def is_terminal(self, state: LifecycleState) -> bool:
        """Check if episode is done.

        Args:
            state: Current lifecycle state

        Returns:
            True if in terminal state (DEPARTED)
        """
        return state.phase == LifecyclePhase.DEPARTED

    # ------------------------------------------------------------------
    # Phase enter/exit methods
    # ------------------------------------------------------------------

    def _enter_phase(self, phase: LifecyclePhase) -> None:
        """Set phase-specific aircraft state on phase entry."""
        if self._state is None:
            return
        aircraft = self._get_primary_aircraft()
        if aircraft:
            aircraft.phase = phase

        if phase == LifecyclePhase.LANDING:
            # Target landing speed
            if aircraft:
                aircraft.speed_kt = 160.0
        elif phase == LifecyclePhase.ARRIVAL_HANDOFF:
            # Slow to taxi speed
            if aircraft:
                aircraft.speed_kt = 20.0
        elif phase == LifecyclePhase.TAXI_IN:
            if aircraft:
                aircraft.speed_kt = 20.0
        elif phase == LifecyclePhase.DOCKING:
            if aircraft:
                aircraft.speed_kt = 0.0
            self._docking_timer = 0.0
        elif phase == LifecyclePhase.AT_GATE:
            if aircraft:
                aircraft.speed_kt = 0.0
            self._turnaround_timer = 0.0
        elif phase == LifecyclePhase.PUSHBACK:
            if aircraft:
                aircraft.speed_kt = 5.0
        elif phase == LifecyclePhase.TAXI_OUT:
            if aircraft:
                aircraft.speed_kt = 20.0
        elif phase == LifecyclePhase.DEPARTURE_QUEUE:
            if aircraft:
                aircraft.speed_kt = 0.0
        elif phase == LifecyclePhase.TAKEOFF:
            if aircraft:
                aircraft.speed_kt = 30.0

    def _exit_phase(
        self, phase: LifecyclePhase, action: Action
    ) -> tuple[bool, list[str]]:
        """Validate action legality for phase exit.

        Returns:
            Tuple of (transition_allowed, issues)
        """
        if self._state is None:
            return False, ["no_state"]

        issues: list[str] = []

        if phase == LifecyclePhase.APPROACH:
            # Can only exit when altitude reaches landing threshold
            aircraft = self._get_primary_aircraft()
            if aircraft and aircraft.altitude_ft > LANDING_ALTITUDE_THRESHOLD_FT:
                issues.append("altitude_not_at_threshold")
                return False, issues
        elif phase == LifecyclePhase.LANDING:
            # Requires valid landing clearance + runway assignment
            if action.clearance_type != ClearanceType.LANDING:
                issues.append("invalid_clearance_for_landing")
                return False, issues
            if not action.runway:
                issues.append("missing_runway_assignment")
                return False, issues
        elif phase == LifecyclePhase.ARRIVAL_HANDOFF:
            # Requires ground frequency confirmation
            ground_confirmed = self._state.metadata.get(
                "ground_frequency_confirmed", False
            )
            if not ground_confirmed:
                issues.append("ground_frequency_not_confirmed")
                return False, issues
        elif phase == LifecyclePhase.TAXI_IN:
            # Requires taxi clearance with valid route
            if action.clearance_type != ClearanceType.TAXI:
                issues.append("invalid_clearance_for_taxi_in")
                return False, issues
            if not action.route:
                issues.append("missing_taxi_route")
                return False, issues
        elif phase == LifecyclePhase.PUSHBACK:
            # Requires pushback_direction
            if not action.pushback_direction:
                issues.append("missing_pushback_direction")
                return False, issues
        elif phase == LifecyclePhase.TAKEOFF:
            # Requires line_up + takeoff clearance
            line_up = self._state.metadata.get("line_up_confirmed", False)
            takeoff = self._state.metadata.get("takeoff_confirmed", False)
            if not line_up or not takeoff:
                issues.append("missing_takeoff_clearance")
                return False, issues

        return True, issues

    # ------------------------------------------------------------------
    # Phase step implementations
    # ------------------------------------------------------------------

    def _step_approach(self, action: Action) -> tuple[float, list[str]]:
        """Handle APPROACH phase step."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Check if altitude has reached landing threshold
        if aircraft.altitude_ft <= LANDING_ALTITUDE_THRESHOLD_FT:
            next_phase = LifecyclePhase.LANDING
            self._state.phase = next_phase
            self._state.completed_phases.append(LifecyclePhase.APPROACH)
            self._enter_phase(next_phase)
            return 0.1, []

        # Continue descent - simplified physics
        descent_rate_fpm = 1500.0  # Standard descent rate
        descent_ft = descent_rate_fpm * DEFAULT_DT_S / 60.0
        aircraft.altitude_ft = max(0.0, aircraft.altitude_ft - descent_ft)

        # Update position on glide path (simplified - moves toward threshold)
        aircraft.y_ft -= 100.0 * DEFAULT_DT_S  # Moving toward runway

        return 0.0, []

    def _step_landing(self, action: Action) -> tuple[float, list[str]]:
        """Handle LANDING phase step."""
        can_exit, issues = self._exit_phase(LifecyclePhase.LANDING, action)
        if not can_exit:
            return 0.0, issues

        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Find runway threshold and update position
        threshold = self._state.metadata.get("threshold_node", "THR_27L")
        threshold_node = self._schema.nodes.get(threshold)
        runway_heading = self._state.metadata.get("runway_heading", 270.0)

        # Only reset to threshold if not already past it
        if threshold_node:
            runway_end_y = threshold_node.y_ft - 2500.0
            if aircraft.y_ft >= threshold_node.y_ft:
                aircraft.x_ft = threshold_node.x_ft
                aircraft.y_ft = threshold_node.y_ft
        else:
            runway_end_y = -2500.0

        aircraft.altitude_ft = 0.0
        aircraft.speed_kt = 0.0
        aircraft.heading_deg = runway_heading

        # Check if aircraft has reached end of runway (for runway exit)
        if aircraft.y_ft <= runway_end_y:
            next_phase = LifecyclePhase.ARRIVAL_HANDOFF
            self._state.phase = next_phase
            self._state.completed_phases.append(LifecyclePhase.LANDING)
            self._enter_phase(next_phase)
            return 0.1, []

        # Continue rollout
        aircraft.y_ft -= 30.0 * DEFAULT_DT_S
        return 0.0, []

    def _step_arrival_handoff(self, action: Action) -> tuple[float, list[str]]:
        """Handle ARRIVAL_HANDOFF phase step."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Ground frequency confirmation happens via action
        if action.clearance_type == ClearanceType.TAXI:
            # Valid handoff action - confirm ground frequency
            self._state.metadata["ground_frequency_confirmed"] = True

        if self._state.metadata.get("ground_frequency_confirmed", False):
            # Transition to taxi_in after handoff complete
            next_phase = LifecyclePhase.TAXI_IN
            self._state.phase = next_phase
            self._state.completed_phases.append(LifecyclePhase.ARRIVAL_HANDOFF)
            self._enter_phase(next_phase)
            return 0.1, []

        return 0.0, []

    def _step_taxi_in(self, action: Action) -> tuple[float, list[str]]:
        """Handle TAXI_IN phase step."""
        can_exit, issues = self._exit_phase(LifecyclePhase.TAXI_IN, action)
        if not can_exit:
            return 0.0, issues

        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Store taxi route
        if action.route:
            self._state.metadata["taxi_route"] = action.route
            self._route_index = 0
            self._current_node = action.route[0] if action.route else None

        # Move along route
        if self._current_node and self._route_index < len(action.route):
            target_node_id = action.route[self._route_index]
            target_node = self._schema.nodes.get(target_node_id)
            if target_node:
                # Move towards target node
                dx = target_node.x_ft - aircraft.x_ft
                dy = target_node.y_ft - aircraft.y_ft
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < 100.0:  # Close enough to node
                    self._route_index += 1
                    if self._route_index >= len(action.route):
                        # Reached end of route - check if at assigned gate
                        gate_node = self._state.metadata.get("gate_node", "GATE_A1")
                        if (
                            target_node_id == gate_node
                            or target_node.node_type.value in ("gate", "stand")
                        ):
                            next_phase = LifecyclePhase.DOCKING
                            self._state.phase = next_phase
                            self._state.completed_phases.append(LifecyclePhase.TAXI_IN)
                            self._enter_phase(next_phase)
                            return 0.1, []
                else:
                    # Move towards node
                    import math

                    target_heading = math.degrees(math.atan2(dx, dy))
                    heading_rad = math.radians(target_heading)
                    aircraft.heading_deg = target_heading
                    speed_kt = 20.0
                    distance_ft = speed_kt * (6076.1 / 3600) * DEFAULT_DT_S
                    aircraft.x_ft += distance_ft * math.sin(heading_rad)
                    aircraft.y_ft += distance_ft * math.cos(heading_rad)

        return 0.0, []

    def _step_docking(self, action: Action) -> tuple[float, list[str]]:
        """Handle DOCKING phase step."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        self._docking_timer += DEFAULT_DT_S
        if self._docking_timer >= DOCKING_DURATION_S:
            next_phase = LifecyclePhase.AT_GATE
            self._state.phase = next_phase
            self._state.completed_phases.append(LifecyclePhase.DOCKING)
            self._enter_phase(next_phase)
            return 0.1, []

        return 0.0, []

    def _step_at_gate(self, action: Action) -> tuple[float, list[str]]:
        """Handle AT_GATE phase step."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Track turnaround timer - increments each step at AT_GATE
        self._turnaround_timer += DEFAULT_DT_S

        if self._turnaround_timer < TURNAROUND_DELAY_S:
            return 0.0, []

        if action.clearance_type == ClearanceType.PUSHBACK:
            if action.pushback_direction:
                next_phase = LifecyclePhase.PUSHBACK
                self._state.phase = next_phase
                self._state.completed_phases.append(LifecyclePhase.AT_GATE)
                self._state.metadata["pushback_direction"] = action.pushback_direction
                self._enter_phase(next_phase)
                return 0.1, []

        return 0.0, []

    def _step_pushback(self, action: Action) -> tuple[float, list[str]]:
        """Handle PUSHBACK phase step."""
        can_exit, issues = self._exit_phase(LifecyclePhase.PUSHBACK, action)
        if not can_exit:
            return 0.0, issues

        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Move in pushback direction
        pushback_dir = self._state.metadata.get("pushback_direction", "back")
        direction_sign = -1.0 if pushback_dir == "back" else 1.0

        # Update position based on heading and direction
        aircraft.y_ft += direction_sign * 10.0 * DEFAULT_DT_S

        # Check if pushback complete (moved enough distance)
        if abs(aircraft.y_ft) > 400.0:
            # Pushback complete - transition to taxi_out
            self._pushback_complete = True
            next_phase = LifecyclePhase.TAXI_OUT
            self._state.phase = next_phase
            self._state.completed_phases.append(LifecyclePhase.PUSHBACK)
            self._enter_phase(next_phase)
            return 0.1, []

        return 0.0, []

    def _step_taxi_out(self, action: Action) -> tuple[float, list[str]]:
        """Handle TAXI_OUT phase step."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Store taxi route to departure queue
        if action.route and not self._state.metadata.get("taxi_route"):
            self._state.metadata["taxi_route"] = action.route
            self._route_index = 0
            self._current_node = action.route[0] if action.route else None

        # Check if already reached end of route
        if action.route and self._route_index >= len(action.route):
            next_phase = LifecyclePhase.DEPARTURE_QUEUE
            self._state.phase = next_phase
            self._state.completed_phases.append(LifecyclePhase.TAXI_OUT)
            self._enter_phase(next_phase)
            return 0.1, []

        # Move along route
        if (
            self._current_node
            and action.route
            and self._route_index < len(action.route)
        ):
            target_node_id = action.route[self._route_index]
            target_node = self._schema.nodes.get(target_node_id)
            if target_node:
                dx = target_node.x_ft - aircraft.x_ft
                dy = target_node.y_ft - aircraft.y_ft
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < 100.0:
                    self._route_index += 1
                    if self._route_index >= len(action.route):
                        # Reached departure queue
                        next_phase = LifecyclePhase.DEPARTURE_QUEUE
                        self._state.phase = next_phase
                        self._state.completed_phases.append(LifecyclePhase.TAXI_OUT)
                        self._enter_phase(next_phase)
                        return 0.1, []
                else:
                    import math

                    target_heading = math.degrees(math.atan2(dx, dy))
                    heading_rad = math.radians(target_heading)
                    aircraft.heading_deg = target_heading
                    speed_kt = 20.0
                    distance_ft = speed_kt * (6076.1 / 3600) * DEFAULT_DT_S
                    aircraft.x_ft += distance_ft * math.sin(heading_rad)
                    aircraft.y_ft += distance_ft * math.cos(heading_rad)

        return 0.0, []

    def _step_departure_queue(self, action: Action) -> tuple[float, list[str]]:
        """Handle DEPARTURE_QUEUE phase step."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Line up and takeoff clearances
        if action.clearance_type == ClearanceType.LINE_UP:
            self._state.metadata["line_up_confirmed"] = True
        elif action.clearance_type == ClearanceType.TAKEOFF:
            self._state.metadata["takeoff_confirmed"] = True

        if self._state.metadata.get(
            "line_up_confirmed", False
        ) and self._state.metadata.get("takeoff_confirmed", False):
            next_phase = LifecyclePhase.TAKEOFF
            self._state.phase = next_phase
            self._state.completed_phases.append(LifecyclePhase.DEPARTURE_QUEUE)
            self._enter_phase(next_phase)
            return 0.1, []

        return 0.0, []

    def _step_takeoff(self, action: Action) -> tuple[float, list[str]]:
        """Handle TAKEOFF phase step."""
        can_exit, issues = self._exit_phase(LifecyclePhase.TAKEOFF, action)
        if not can_exit:
            return 0.0, issues

        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return 0.0, ["no_aircraft"]

        # Accelerate and takeoff
        aircraft.speed_kt = min(aircraft.speed_kt + 20.0 * DEFAULT_DT_S, 160.0)
        aircraft.y_ft -= aircraft.speed_kt * DEFAULT_DT_S * 0.5  # Moving forward

        # Check if aircraft has left runway (y position past runway end)
        threshold = self._state.metadata.get("threshold_node", "THR_27L")
        threshold_node = self._schema.nodes.get(threshold)
        runway_length = 9850.0  # From gatwick schema
        if threshold_node:
            if aircraft.y_ft < threshold_node.y_ft - runway_length:
                next_phase = LifecyclePhase.DEPARTED
                self._state.phase = next_phase
                self._state.completed_phases.append(LifecyclePhase.TAKEOFF)
                self._state.completed_phases.append(LifecyclePhase.DEPARTED)
                self._enter_phase(next_phase)
                return 0.1, []

        return 0.0, []

    # ------------------------------------------------------------------
    # Legal action helpers
    # ------------------------------------------------------------------

    def _legal_approach_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for APPROACH phase."""
        aircraft = self._get_primary_aircraft()
        if aircraft and aircraft.altitude_ft <= LANDING_ALTITUDE_THRESHOLD_FT:
            return [
                Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign=aircraft.callsign,
                    runway=state.metadata.get("assigned_runway", "27L"),
                )
            ]
        return []

    def _legal_landing_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for LANDING phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        return [
            Action(
                clearance_type=ClearanceType.LANDING,
                target_callsign=aircraft.callsign,
                runway=state.metadata.get("assigned_runway", "27L"),
            )
        ]

    def _legal_arrival_handoff_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for ARRIVAL_HANDOFF phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        if not state.metadata.get("ground_frequency_confirmed", False):
            return [
                Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign=aircraft.callsign,
                    route=[],
                )
            ]
        return []

    def _legal_taxi_in_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for TAXI_IN phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        # Build route to assigned gate
        gate_node = state.metadata.get("gate_node", "GATE_A1")
        return [
            Action(
                clearance_type=ClearanceType.TAXI,
                target_callsign=aircraft.callsign,
                route=[gate_node],
            )
        ]

    def _legal_docking_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for DOCKING phase."""
        # Docking is automated - no actions required
        return []

    def _legal_at_gate_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for AT_GATE phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        return [
            Action(
                clearance_type=ClearanceType.PUSHBACK,
                target_callsign=aircraft.callsign,
                pushback_direction="back",
            )
        ]

    def _legal_pushback_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for PUSHBACK phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        direction = state.metadata.get("pushback_direction", "back")
        return [
            Action(
                clearance_type=ClearanceType.PUSHBACK,
                target_callsign=aircraft.callsign,
                pushback_direction=direction,
            )
        ]

    def _legal_taxi_out_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for TAXI_OUT phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        # Build route to departure queue
        return [
            Action(
                clearance_type=ClearanceType.TAXI,
                target_callsign=aircraft.callsign,
                route=["DQ_E", "RE_E"],  # Simplified route to departure queue
            )
        ]

    def _legal_departure_queue_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for DEPARTURE_QUEUE phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        actions = []
        if not state.metadata.get("line_up_confirmed", False):
            actions.append(
                Action(
                    clearance_type=ClearanceType.LINE_UP,
                    target_callsign=aircraft.callsign,
                )
            )
        if state.metadata.get("line_up_confirmed", False) and not state.metadata.get(
            "takeoff_confirmed", False
        ):
            actions.append(
                Action(
                    clearance_type=ClearanceType.TAKEOFF,
                    target_callsign=aircraft.callsign,
                )
            )
        return actions

    def _legal_takeoff_actions(self, state: LifecycleState) -> list[Action]:
        """Return legal actions for TAKEOFF phase."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return []
        return [
            Action(
                clearance_type=ClearanceType.TAKEOFF,
                target_callsign=aircraft.callsign,
            )
        ]

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_state(self) -> LifecycleState:
        """Get state with assertion that it is not None."""
        assert self._state is not None, "State machine not initialized"
        return self._state

    def _get_primary_aircraft(self) -> AircraftState | None:
        """Get the primary (first) aircraft from state."""
        if self._state and self._state.aircraft_states:
            return next(iter(self._state.aircraft_states.values()))
        return None

    def _find_first_node_by_type(self, node_type: str) -> Any:
        """Find first node of given type in schema."""
        for node in self._schema.nodes.values():
            if node.node_type.value == node_type:
                return node
        return None

    def _update_aircraft_physics(self, dt_s: float) -> None:
        """Update aircraft position/heading based on phase and physics."""
        aircraft = self._get_primary_aircraft()
        if not aircraft:
            return

        phase = self._state.phase if self._state else None
        if phase in (
            LifecyclePhase.DOCKING,
            LifecyclePhase.AT_GATE,
            LifecyclePhase.DEPARTED,
        ):
            # Stationary
            aircraft.speed_kt = 0.0
        elif phase in (LifecyclePhase.ARRIVAL_HANDOFF,):
            # Slowing to taxi speed
            if aircraft.speed_kt > 20.0:
                aircraft.speed_kt = max(aircraft.speed_kt - 10.0 * dt_s, 20.0)
