"""HFT Oversight Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import OversightAction, OversightObservation


class HFTOversightEnv(EnvClient[OversightAction, OversightObservation]):
    """
    Client for the HFT Oversight Environment.

    Example:
        >>> with HFTOversightEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.response)
        ...     result = env.step(OversightAction(command="read_logs", bot_id="citadel"))
        ...     print(result.observation.response)
    """

    def _step_payload(self, action: OversightAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[OversightObservation]:
        obs_data = payload.get("observation", {})
        observation = OversightObservation(
            response=obs_data.get("response", ""),
            timestep=obs_data.get("timestep", 0),
            max_timesteps=obs_data.get("max_timesteps", 20),
            alerts=obs_data.get("alerts", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
