"""FastAPI application for the HFT Oversight Environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from models import OversightAction, OversightObservation
from .environment import HFTOversightEnvironment

app = create_app(
    HFTOversightEnvironment,
    OversightAction,
    OversightObservation,
    env_name="hft_oversight",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
