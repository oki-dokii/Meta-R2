"""
server/app.py — LifeStack OpenEnv entry point.

This module is the canonical server entry point registered in pyproject.toml
as [project.scripts] server = "server.app:main".  It wraps LifeStackEnv in an
HTTP + WebSocket server compatible with EnvClient / openenv validate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path when invoked as `python server/app.py`
# (it already is when invoked as `server` console script via pyproject.toml).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    """
    LifeStack OpenEnv Server — standard distribution entry point.
    Wraps LifeStackEnv in an HTTP and WebSocket server compatible with EnvClient.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is not installed.  Run: pip install uvicorn"
        ) from exc

    try:
        from openenv.core import create_app
    except Exception as exc:
        raise SystemExit(
            f"openenv.core could not be imported ({exc}).  "
            "Ensure openenv-core>=0.2.3, fastapi, and fastmcp are installed."
        ) from exc

    from core.lifestack_env import LifeStackEnv, LifeStackAction, LifeStackObservation, USING_MODERN_API

    if not USING_MODERN_API:
        raise SystemExit(
            "openenv.core modern API is not available (USING_MODERN_API=False).  "
            "Check that openenv-core>=0.2.3 and its dependencies (fastapi, fastmcp) "
            "are correctly installed in this Python environment."
        )

    host = os.getenv("OPENENV_HOST", "0.0.0.0")
    port = int(os.getenv("OPENENV_PORT", "8000"))
    max_concurrent = int(os.getenv("OPENENV_MAX_SESSIONS", "4"))

    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

    app = create_app(
        env=LifeStackEnv,
        action_cls=LifeStackAction,
        observation_cls=LifeStackObservation,
        env_name="LifeStack Premium",
        max_concurrent_envs=max_concurrent,
    )

    print("\n" + "=" * 60)
    print("  LifeStack OpenEnv Server is ready!")
    print(f"  HTTP Endpoint : http://{host}:{port}")
    print(f"  Web Interface : http://{host}:{port}/web")
    print(f"  Documentation : http://{host}:{port}/docs")
    print(f"  Health check  : http://{host}:{port}/health")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
