import os
import uvicorn
try:
    from openenv.core import create_app
except ImportError:
    # Alternative path for older/flat openenv structures
    try:
        from openenv.env import create_app
    except ImportError:
        def create_app(*a, **k):
            print("⚠️ create_app not found in openenv. Using fallback.")
            return None
from lifestack_env import LifeStackEnv, LifeStackAction, LifeStackObservation

def main():
    """
    LifeStack OpenEnv Server — Standard distribution entry point.
    Wraps LifeStackEnv in an HTTP and WebSocket server compatible with EnvClient.
    """
    # Use standard environment variables for configuration
    host = os.getenv("OPENENV_HOST", "0.0.0.0")
    port = int(os.getenv("OPENENV_PORT", "8000"))
    max_concurrent = int(os.getenv("OPENENV_MAX_SESSIONS", "4"))
    
    # Create the FastAPI app with the builtin OpenEnv web interface enabled
    os.environ["ENABLE_WEB_INTERFACE"] = "true" 
    
    app = create_app(
        env=LifeStackEnv,
        action_cls=LifeStackAction,
        observation_cls=LifeStackObservation,
        env_name="LifeStack Premium",
        max_concurrent_envs=max_concurrent
    )
    
    print(f"\n" + "═"*60)
    print(f"  🚀 LifeStack OpenEnv Server is ready!")
    print(f"  - HTTP Endpoint: http://{host}:{port}")
    print(f"  - Web Interface: http://{host}:{port}/web")
    print(f"  - Documentation: http://{host}:{port}/docs")
    print("═"*60 + "\n")
    
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
