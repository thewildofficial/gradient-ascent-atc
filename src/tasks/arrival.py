"""Arrival task: landing + handoff + taxi-in + docking."""

from pydantic import BaseModel, ConfigDict, Field

from src.airport_schema import AirportSchema
from src.models import Action, LifecyclePhase, Observation
from src.rewards import RewardCalculator
from src.state_machine import FullLifecycleStateMachine, LifecycleState


ARRIVAL_PHASES: list[LifecyclePhase] = [
    LifecyclePhase.APPROACH,
    LifecyclePhase.LANDING,
    LifecyclePhase.ARRIVAL_HANDOFF,
    LifecyclePhase.TAXI_IN,
    LifecyclePhase.DOCKING,
    LifecyclePhase.AT_GATE,
]

COMPLETION_BONUS = 0.1
COMPLETION_BONUS_PHASES = 6


class ArrivalTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    airport_schema: AirportSchema
    seed: int | None = None
    state_machine: FullLifecycleStateMachine = Field(default=None, exclude=True)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        object.__setattr__(
            self,
            "state_machine",
            FullLifecycleStateMachine(schema=self.airport_schema, seed=self.seed),
        )

    def reset(self, episode_id: str) -> LifecycleState:
        return self.state_machine.reset(task_id="arrival", episode_id=episode_id)

    def step(self, action: Action) -> tuple[LifecycleState, Observation]:
        return self.state_machine.step(action)

    def is_terminal(self, state: LifecycleState) -> bool:
        return state.phase == LifecyclePhase.AT_GATE


class ArrivalGrader:
    def __init__(self) -> None:
        self.reward_calculator = RewardCalculator()

    def grade(self, state: LifecycleState, rewards: list[float]) -> float:
        if not rewards:
            return 0.0

        mean_reward = sum(rewards) / len(rewards)

        bonus = 0.0
        if self._all_arrival_phases_completed(state):
            bonus = COMPLETION_BONUS

        total = mean_reward + bonus

        return max(0.0, min(1.0, total))

    def _all_arrival_phases_completed(self, state: LifecycleState) -> bool:
        completed = set(state.completed_phases)
        required = set(ARRIVAL_PHASES)
        return required.issubset(completed)
