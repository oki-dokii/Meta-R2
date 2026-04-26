import os


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        print(f"[openenv-server] uvicorn not installed: {exc}. Skipping OpenEnv server.")
        return

    try:
        from openenv.core import create_app
    except Exception as exc:
        print(f"[openenv-server] openenv.core unavailable ({exc}). Skipping OpenEnv server.")
        return

    from core.lifestack_env import LifeStackEnv, LifeStackAction, LifeStackObservation

    host = os.getenv("OPENENV_HOST", "0.0.0.0")
    port = int(os.getenv("OPENENV_PORT", "8000"))
    max_concurrent = int(os.getenv("OPENENV_MAX_SESSIONS", "4"))

    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

    try:
        app = create_app(
            env=LifeStackEnv,
            action_cls=LifeStackAction,
            observation_cls=LifeStackObservation,
            env_name="LifeStack Premium",
            max_concurrent_envs=max_concurrent,
        )
    except Exception as exc:
        print(f"[openenv-server] create_app failed ({exc}). Skipping OpenEnv server.")
        return

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
