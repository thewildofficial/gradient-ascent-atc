"""Reward engine and grader primitives."""

from pydantic import BaseModel, Field

from src.models import Action, Observation
from src.state_machine import LifecycleState


TOTAL_PHASES = 11
OPTIMAL_STEPS = 11


class RewardSignal(BaseModel):
    """Signal components for reward breakdown."""

    model_config = {"extra": "forbid"}

    safety: float = Field(..., ge=0.0, le=1.0)
    legality: float = Field(..., ge=0.0, le=1.0)
    completion: float = Field(..., ge=0.0, le=1.0)
    efficiency: float = Field(..., ge=0.0, le=1.0)
    communication: float = Field(..., ge=0.0, le=1.0)


class RewardCalculator:
    """Evaluates agent decisions with safety, legality, completion, and efficiency scoring."""

    weights: dict[str, float] = {
        "safety": 0.4,
        "legality": 0.25,
        "completion": 0.2,
        "efficiency": 0.1,
        "communication": 0.05,
    }

    def compute_reward(
        self, state: LifecycleState, action: Action, obs: Observation
    ) -> tuple[RewardSignal, float]:
        """Compute reward signal and total weighted score.

        Args:
            state: Current lifecycle state
            action: Action taken by agent
            obs: Observation returned by environment

        Returns:
            Tuple of (reward signal components, total weighted score clamped to [0.0, 1.0])
        """
        safety = self._compute_safety(state, obs)
        legality = self._compute_legality(state, action, obs)
        completion = self._compute_completion(state)
        efficiency = self._compute_efficiency(state)
        communication = self._compute_communication(obs)

        signal = RewardSignal(
            safety=safety,
            legality=legality,
            completion=completion,
            efficiency=efficiency,
            communication=communication,
        )

        total = self._compute_weighted_total(signal, safety)

        return signal, total

    def _compute_safety(self, state: LifecycleState, obs: Observation) -> float:
        """Compute safety score: +1.0 if no violations, +0.0 if collision/runway incursion."""
        if "collision" in obs.issues or "runway_incursion" in obs.issues:
            return 0.0
        if state.metadata.get("safety_violation"):
            return 0.0
        return 1.0

    def _compute_legality(
        self, state: LifecycleState, action: Action, obs: Observation
    ) -> float:
        """Compute legality score: +1.0 if valid, +0.0 if protocol violations."""
        if "protocol_violation" in obs.issues or "invalid_clearance" in obs.issues:
            return 0.0
        if obs.result == "illegal_transition":
            return 0.0
        return 1.0

    def _compute_completion(self, state: LifecycleState) -> float:
        """Compute completion score as fraction of phases completed / total phases."""
        completed = len(state.completed_phases)
        return completed / TOTAL_PHASES

    def _compute_efficiency(self, state: LifecycleState) -> float:
        """Compute efficiency score: inverse of steps taken vs optimal."""
        steps = max(state.step_count, 1)
        return min(OPTIMAL_STEPS / steps, 1.0)

    def _compute_communication(self, obs: Observation) -> float:
        """Compute communication score from phraseology correctness."""
        return 1.0 if obs.phraseology_ok else 0.0

    def _compute_weighted_total(self, signal: RewardSignal, safety: float) -> float:
        """Compute weighted total, with safety dominance (safety=0 → total=0)."""
        if safety == 0.0:
            return 0.0

        total = (
            signal.safety * self.weights["safety"]
            + signal.legality * self.weights["legality"]
            + signal.completion * self.weights["completion"]
            + signal.efficiency * self.weights["efficiency"]
            + signal.communication * self.weights["communication"]
        )

        return max(0.0, min(1.0, total))


class TaskGrader:
    """Aggregates episode rewards and returns final normalized score."""

    def grade_episode(self, episode_rewards: list[float], task_id: str) -> float:
        """Aggregate rewards over episode and return final score.

        Args:
            episode_rewards: List of per-step reward values
            task_id: Task identifier (unused in base grader)

        Returns:
            Final score in [0.0, 1.0]
        """
        if not episode_rewards:
            return 0.0

        total = sum(episode_rewards)
        avg = total / len(episode_rewards)

        return max(0.0, min(1.0, avg))
