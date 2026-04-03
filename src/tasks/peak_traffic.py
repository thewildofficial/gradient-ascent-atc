"""Peak traffic task — 3 aircraft simultaneously, collision avoidance required."""

from pydantic import BaseModel, ConfigDict

from src.airport_schema import AirportSchema
from src.models import Action, LifecyclePhase, Observation
from src.physics import check_all_collisions
from src.state_machine import FullLifecycleStateMachine, LifecycleState


PEAK_TRAFFIC_AIRCRAFT_COUNT = 3
COMPLETION_BONUS_PER_AIRCRAFT = 0.05
COLLISION_PENALTY = -1.0


class PeakTrafficTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    airport_schema: AirportSchema
    seed: int | None = None
    state_machines: dict[str, FullLifecycleStateMachine] | None = None

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.state_machines = {}
        callsigns = ["BAW456", "EZY789", "AFR321"]
        for callsign in callsigns:
            self.state_machines[callsign] = FullLifecycleStateMachine(
                schema=self.airport_schema, seed=self.seed
            )

    def reset(self, episode_id: str) -> LifecycleState:
        self.state_machines["BAW456"].reset(
            task_id="peak_traffic", episode_id=f"{episode_id}-BAW456"
        )
        self.state_machines["EZY789"].reset(
            task_id="peak_traffic", episode_id=f"{episode_id}-EZY789"
        )
        self.state_machines["AFR321"].reset(
            task_id="peak_traffic", episode_id=f"{episode_id}-AFR321"
        )
        return self._combined_state()

    def step(self, action: Action) -> tuple[LifecycleState, Observation]:
        collisions = check_all_collisions(self._get_all_aircraft_states())
        if collisions:
            return self._combined_state(), Observation(
                result="collision_detected",
                score=COLLISION_PENALTY,
                phraseology_ok=False,
                issues=[f"collision between {c[0]} and {c[1]}" for c in collisions],
            )

        machine = self.state_machines.get(action.target_callsign)
        if machine:
            return machine.step(action)

        return self._combined_state(), Observation(
            result="unknown_callsign",
            score=0.0,
            phraseology_ok=False,
            issues=[f"unknown callsign: {action.target_callsign}"],
        )

    def is_terminal(self, state: LifecycleState) -> bool:
        return all(m.is_terminal(m._state) for m in self.state_machines.values())

    def _combined_state(self) -> LifecycleState:
        combined_aircraft = {}
        for callsign, machine in self.state_machines.items():
            combined_aircraft.update(machine._state.aircraft_states)

        primary_state = next(iter(self.state_machines.values()))._state
        return LifecycleState(
            phase=primary_state.phase,
            aircraft_states=combined_aircraft,
            episode_id=primary_state.episode_id,
            step_count=primary_state.step_count,
            task_id="peak_traffic",
            completed_phases=primary_state.completed_phases,
        )

    def _get_all_aircraft_states(self):
        states = {}
        for callsign, machine in self.state_machines.items():
            states.update(machine._state.aircraft_states)
        return states


class PeakTrafficGrader:
    def __init__(self) -> None:
        self._collision_occurred = False

    def grade(self, state: LifecycleState, rewards: list[float]) -> float:
        if not rewards:
            return 0.0

        mean_reward = sum(rewards) / len(rewards)

        if any(
            "collision" in str(issue).lower()
            for issue in state.metadata.get("issues", [])
        ):
            self._collision_occurred = True
            return COLLISION_PENALTY

        bonus = 0.0
        completed_aircraft = 0
        for phase in [LifecyclePhase.DEPARTED]:
            if state.completed_phases.count(phase) >= PEAK_TRAFFIC_AIRCRAFT_COUNT:
                completed_aircraft = PEAK_TRAFFIC_AIRCRAFT_COUNT
                break

        bonus = completed_aircraft * COMPLETION_BONUS_PER_AIRCRAFT

        total = mean_reward + bonus
        return max(-1.0, min(1.0, total))
