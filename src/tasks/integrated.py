"""Integrated full-lifecycle task: landing through departure."""

from pydantic import BaseModel, Field

from src.models import LifecyclePhase
from src.rewards import RewardCalculator, TaskGrader
from src.state_machine import LifecycleState


FULL_PHASE_COUNT = 11
PHASE_PENALTY = 0.05
COMPLETION_BONUS = 0.2


class IntegratedTask(BaseModel):
    """End-to-end full lifecycle episode combining arrival and departure.

    Phase sequence: APPROACH → LANDING → ARRIVAL_HANDOFF → TAXI_IN →
    DOCKING → AT_GATE → PUSHBACK → TAXI_OUT → DEPARTURE_QUEUE →
    TAKEOFF → DEPARTED

    Turnaround delay: 60 seconds at AT_GATE before pushback is allowed.
    """

    model_config = {"extra": "forbid"}

    turnaround_delay_s: float = Field(default=60.0, ge=0.0)


class IntegratedGrader:
    """Grader for integrated full-lifecycle task.

    Uses RewardCalculator from src.rewards and applies additional
    phase penalties and completion bonuses specific to the integrated task.
    """

    def __init__(self) -> None:
        self._reward_calculator = RewardCalculator()
        self._base_grader = TaskGrader()

    def grade(self, state: LifecycleState, rewards: list[float]) -> float:
        """Calculate final score for an integrated task episode.

        Args:
            state: Final LifecycleState of the episode
            rewards: List of per-step reward values

        Returns:
            Final score in [0.0, 1.0]
        """
        if not rewards:
            return 0.0

        base_score = self._base_grader.grade_episode(rewards, "integrated")

        completed = set(state.completed_phases)
        expected_phases = set(LifecyclePhase)

        missing_phases = expected_phases - completed
        phase_penalty = len(missing_phases) * PHASE_PENALTY

        if len(completed_phases_from_state(state)) == FULL_PHASE_COUNT:
            final_score = base_score + COMPLETION_BONUS
        else:
            final_score = base_score - phase_penalty

        return max(0.0, min(1.0, final_score))


def completed_phases_from_state(state: LifecycleState) -> list[LifecyclePhase]:
    """Extract completed phases list from state.

    Args:
        state: LifecycleState to extract from

    Returns:
        List of completed phases
    """
    return state.completed_phases
