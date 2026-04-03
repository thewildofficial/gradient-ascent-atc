"""Benchmark module for running ATC Ground Control RL tasks."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from src.models import Action, ClearanceType
from src.openenv_environment import OpenEnvEnvironment
from src.tasks.arrival import ArrivalGrader
from src.tasks.departure import DepartureGrader
from src.tasks.integrated import IntegratedGrader
from src.tasks.peak_traffic import PeakTrafficGrader
from src.tasks.registry import TaskRegistry


def list_tasks() -> list[dict[str, Any]]:
    tasks = TaskRegistry.list_tasks()
    return [
        {"task_id": t.task_id, "name": t.name, "difficulty": t.difficulty}
        for t in tasks
    ]


def _get_grader(task_id: str) -> Any:
    if task_id == "arrival":
        return ArrivalGrader()
    elif task_id == "departure":
        return DepartureGrader()
    elif task_id == "integrated":
        return IntegratedGrader()
    elif task_id == "peak_traffic":
        return PeakTrafficGrader()
    raise KeyError(f"Unknown task: {task_id}")


async def _run_episode_async(task_id: str, seed: int) -> tuple[float, list[float]]:
    env = OpenEnvEnvironment(task_id=task_id, seed=seed)
    rewards: list[float] = []

    try:
        obs = await env.reset()
        rewards.append(obs.score)

        rng = random.Random(seed)
        for _ in range(1000):
            machine = env._machine
            legal_actions = machine.get_legal_actions(machine._state)

            if not legal_actions:
                action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign="BAW123",
                    route=[],
                )
            else:
                action = rng.choice(legal_actions)

            obs, reward, done = await env.step(action)
            rewards.append(reward)

            is_terminal = machine.is_terminal(machine._state)
            has_illegal = "illegal_transition" in obs.issues

            if done or is_terminal or has_illegal:
                break
    finally:
        env.close()

    grader = _get_grader(task_id)
    machine = env._machine if env._lifecycle_state else None

    if machine is not None and machine._state is not None:
        final_state = machine._state
    else:
        from src.state_machine import LifecycleState
        from src.models import LifecyclePhase

        final_state = LifecycleState(
            phase=LifecyclePhase.DEPARTED, completed_phases=list(LifecyclePhase)
        )

    return grader.grade(final_state, rewards), rewards


def run_task(task_id: str, seed: int) -> dict[str, Any]:
    TaskRegistry.get(task_id)
    final_score, rewards = asyncio.run(_run_episode_async(task_id, seed))
    return {
        "task_id": task_id,
        "seed": seed,
        "score": max(0.0, min(1.0, final_score)),
        "rewards": rewards,
    }


def run_all() -> dict[str, float]:
    scores: dict[str, float] = {}
    for task in TaskRegistry.list_tasks():
        result = run_task(task.task_id, hash(task.task_id) % 10000)
        scores[task.task_id] = result["score"]
    return scores


if __name__ == "__main__":
    print("Available tasks:")
    for task in list_tasks():
        print(f"  - {task['task_id']}: {task['name']} ({task['difficulty']})")
    print("\nRunning all tasks...")
    scores = run_all()
    print("\nScores:")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.3f}")
