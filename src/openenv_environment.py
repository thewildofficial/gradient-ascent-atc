"""OpenEnv-compatible environment wrapper."""

import uuid

from src.airport_schema import AirportSchemaLoader
from src.models import Action, Observation, State
from src.state_machine import FullLifecycleStateMachine


class OpenEnvEnvironment:
    def __init__(self, task_id: str, seed: int | None = None) -> None:
        self.task_id = task_id
        self.seed_value = seed
        schema = AirportSchemaLoader.load("dummy_small")
        self._machine = FullLifecycleStateMachine(schema=schema, seed=seed)
        self._lifecycle_state = None

    async def reset(self) -> Observation:
        episode_id = str(uuid.uuid4())
        self._lifecycle_state = self._machine.reset(
            task_id=self.task_id, episode_id=episode_id
        )
        return Observation(
            result="reset complete",
            score=0.0,
            phraseology_ok=True,
            issues=[],
        )

    async def step(self, action: Action) -> tuple[Observation, float, bool]:
        lifecycle_state, obs = self._machine.step(action)
        self._lifecycle_state = lifecycle_state
        done = self._machine.is_terminal(lifecycle_state)
        return obs, obs.score, done

    def state(self) -> State:
        ls = self._lifecycle_state
        return State(
            phase=ls.phase,
            aircraft=ls.aircraft_states,
            episode_id=ls.episode_id,
            step_count=ls.step_count,
            task_id=ls.task_id,
            metadata=ls.metadata,
        )

    def close(self) -> None:
        self._lifecycle_state = None
