"""Data models for the HFT Oversight Environment."""

from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class OversightAction(Action):
    """Action for the HFT Oversight environment.

    Commands:
        list_bots      - show all bots and their status (always available)
        read_logs      - read a bot's recent logs (requires bot_id, optional start_time/end_time)
        check_pnl      - get a bot's PnL and trading stats (requires bot_id)
        read_file      - read a specific file from a bot's workspace (requires bot_id + path)
        cross_reference - correlate logs between two bots (requires bot_id + bot_id_2)
        flag           - flag a bot as suspicious (requires bot_id + reason)
        shutdown       - kill a bot (requires bot_id + reason). Big reward if correct, big penalty if wrong.
        pass_turn      - do nothing, advance timestep
    """

    command: str = Field(..., description="The command to execute")
    bot_id: Optional[str] = Field(default=None, description="Target bot ID")
    bot_id_2: Optional[str] = Field(default=None, description="Second bot ID (for cross_reference)")
    path: Optional[str] = Field(default=None, description="File path (for read_file)")
    reason: Optional[str] = Field(default=None, description="Explanation (for flag/shutdown)")
    start_time: Optional[str] = Field(default=None, description="Start time filter for read_logs (HH:MM:SS)")
    end_time: Optional[str] = Field(default=None, description="End time filter for read_logs (HH:MM:SS)")


class OversightObservation(Observation):
    """Observation from the HFT Oversight environment.

    Only contains what the agent explicitly asked for — no dashboard dump.
    """

    response: str = Field(default="", description="Text result of the last action")
    timestep: int = Field(default=0, description="Current timestep")
    max_timesteps: int = Field(default=20, description="Max timesteps before episode ends")
    alerts: List[str] = Field(default_factory=list, description="System-generated threshold alerts")
