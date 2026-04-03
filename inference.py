"""Competition inference script with exact stdout logging format."""

import asyncio
import os
import random
import sys
from typing import Any

from openai import OpenAI

from src.models import Action, ClearanceType
from src.openenv_environment import OpenEnvEnvironment


def _validate_env_vars() -> None:
    """Validate required environment variables are set."""
    missing = []
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def _get_env(var: str) -> str:
    """Get environment variable value."""
    return os.environ[var]


class RandomAgent:
    """Deterministic baseline agent that picks legal actions at random."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def select_action(self, legal_actions: list[Action]) -> Action | None:
        """Select a random legal action.

        Args:
            legal_actions: List of valid actions for current state.

        Returns:
            Selected Action, or None if no legal actions available (agent should wait).
        """
        if not legal_actions:
            return None
        return self._rng.choice(legal_actions)


def _format_action(action: Action) -> str:
    """Format action as a compact string for logging.

    Args:
        action: Action to format.

    Returns:
        Compact string representation of the action.
    """
    parts = [action.clearance_type.value, action.target_callsign]
    if action.route:
        parts.append(",".join(action.route[:2]))
    if action.pushback_direction:
        parts.append(action.pushback_direction)
    return "|".join(parts)


async def run_episode(
    task_id: str,
    model_name: str,
    agent: RandomAgent,
) -> tuple[bool, int, float, list[float]]:
    """Run a single episode for a task.

    Args:
        task_id: Task identifier (departure, arrival, integrated).
        model_name: Model name for logging.
        agent: Agent to use for action selection.

    Returns:
        Tuple of (success, step_count, total_score, list_of_rewards).
    """
    print(f"[START] task={task_id} env=gradient-ascent-atc model={model_name}")

    env = OpenEnvEnvironment(task_id=task_id, seed=42)
    step_count = 0
    rewards: list[float] = []
    success = False
    error_msg: str | None = None

    try:
        # Reset environment - must use await per requirements
        observation = await env.reset()
        rewards.append(observation.score)

        # Run episode loop
        while True:
            step_count += 1

            # Get legal actions for current state
            state = env.state()
            machine = env._machine
            legal_actions = machine.get_legal_actions(machine._state)

            # Select action - None means no legal actions available, wait
            action = agent.select_action(legal_actions)
            if action is None:
                # No legal actions - create a no-op action to continue
                action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign="BAW123",
                    route=[],
                )
            action_str = _format_action(action)

            # Execute step
            try:
                obs, reward, done = await env.step(action)
                rewards.append(reward)

                # Check for terminal conditions
                machine = env._machine
                is_terminal = machine.is_terminal(machine._state)
                has_illegal_transition = "illegal_transition" in obs.issues

                # Terminal conditions: done flag OR terminal state OR illegal transition
                if done or is_terminal or has_illegal_transition:
                    if has_illegal_transition:
                        error_str = "illegal_transition"
                        success = False
                    else:
                        error_str = "null"
                        success = True
                    reward_str = f"{reward:.2f}"
                    print(
                        f"[STEP] step={step_count} action={action_str} "
                        f"reward={reward_str} done=true error={error_str}"
                    )
                    break

                # Format reward with exactly 2 decimal places
                reward_str = f"{reward:.2f}"

                # Format error - use null (not None)
                error_str = "null"

                print(
                    f"[STEP] step={step_count} action={action_str} "
                    f"reward={reward_str} done={str(done).lower()} error={error_str}"
                )

                # Safety check for runaway episodes
                if step_count >= 1000:
                    error_msg = "max_steps_exceeded"
                    print(
                        f"[STEP] step={step_count} action={action_str} "
                        f"reward={reward_str} done=true error=max_steps_exceeded"
                    )
                    success = False
                    break

            except Exception as e:
                error_msg = str(e)
                print(
                    f"[STEP] step={step_count} action={action_str} "
                    f"reward=0.00 done=true error={error_msg}"
                )
                success = False
                break

    except Exception as e:
        error_msg = str(e)
        success = False
    finally:
        env.close()

    # Calculate total score (average of rewards)
    total_score = sum(rewards) / len(rewards) if rewards else 0.0
    total_score = max(0.0, min(1.0, total_score))  # Clamp to [0.0, 1.0]

    # Format rewards list as comma-separated with 2 decimal places
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={total_score:.3f} rewards={rewards_str}"
    )

    return success, step_count, total_score, rewards


async def main() -> int:
    """Main entry point for inference script."""
    try:
        # Validate environment variables
        _validate_env_vars()

        model_name = _get_env("MODEL_NAME")
        api_base_url = _get_env("API_BASE_URL")
        hf_token = _get_env("HF_TOKEN")

        # Initialize OpenAI client (base_url and api_key as specified)
        client = OpenAI(api_key=hf_token, base_url=api_base_url)

        # Tasks to run
        task_ids = ["departure", "arrival", "integrated"]

        # Initialize deterministic baseline agent
        agent = RandomAgent(seed=42)

        # Run episodes sequentially
        for task_id in task_ids:
            await run_episode(task_id, model_name, agent)

        return 0

    except EnvironmentError as e:
        # Emit [END] to stdout for failure case
        print("[END] success=false steps=0 score=0.000 rewards=")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # Emit [END] to stdout for failure case
        print("[END] success=false steps=0 score=0.000 rewards=")
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
