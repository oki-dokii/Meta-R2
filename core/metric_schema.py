
from core.life_state import LifeMetrics


VALID_METRIC_PATHS = tuple(sorted(LifeMetrics().flatten().keys()))

LEGACY_METRIC_ALIASES = {
    "physical_health.exercise_routine": "physical_health.fitness",
}


def normalize_metric_path(path: str) -> str:
    """Map legacy or malformed metric names onto the current LifeMetrics schema."""
    if not isinstance(path, str):
        return ""
    path = path.strip()
    return LEGACY_METRIC_ALIASES.get(path, path)


def is_valid_metric_path(path: str) -> bool:
    return normalize_metric_path(path) in VALID_METRIC_PATHS


def format_valid_metrics() -> str:
    grouped = {}
    for path in VALID_METRIC_PATHS:
        domain, metric = path.split(".", 1)
        grouped.setdefault(domain, []).append(metric)
    return "\n".join(
        f"{domain}: {', '.join(metrics)}" for domain, metrics in grouped.items()
    )
