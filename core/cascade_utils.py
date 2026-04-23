import copy
from core.life_state import LifeMetrics, DependencyGraph, CASCADE_DAMPENING_DEFAULT


def animate_cascade(primary_disruption: dict, metrics: LifeMetrics) -> list[dict]:
    """Replay the cascade step-by-step and capture intermediate frames.

    Returns a list of frames, each:
      { 'flat': {metric: value}, 'status': {metric: 'primary'|'first'|'second'|'unchanged'} }
    """
    graph = DependencyGraph()
    dampening = CASCADE_DAMPENING_DEFAULT
    frames = []

    # Frame 0 — initial stable state
    base = copy.deepcopy(metrics)
    base_flat = base.flatten()
    frames.append({'flat': dict(base_flat), 'status': {k: 'unchanged' for k in base_flat}})

    # Frame 1 — primary disruption only (no cascade)
    f1 = copy.deepcopy(metrics)
    primary_keys = set()
    for path, amount in primary_disruption.items():
        if '.' not in path:
            continue
        primary_keys.add(path)
        dom_name, sub_name = path.split('.', 1)
        dom = getattr(f1, dom_name, None)
        if dom and hasattr(dom, sub_name):
            setattr(dom, sub_name, max(0.0, min(100.0, getattr(dom, sub_name) + amount)))
    f1_flat = f1.flatten()
    frames.append({'flat': dict(f1_flat),
                   'status': {k: ('primary' if k in primary_keys else 'unchanged') for k in f1_flat}})

    # Frame 2 — first-order cascade
    f2 = copy.deepcopy(f1)
    first_order_keys = set()
    queue_next = []
    for path, amount in primary_disruption.items():
        if '.' not in path or path not in graph.edges:
            continue
        for target, weight in graph.edges[path]:
            impact = amount * weight * dampening
            if abs(impact) >= 0.05:
                first_order_keys.add(target)
                dom_name, sub_name = target.split('.', 1)
                dom = getattr(f2, dom_name, None)
                if dom and hasattr(dom, sub_name):
                    setattr(dom, sub_name, max(0.0, min(100.0, getattr(dom, sub_name) + impact)))
                queue_next.append((target, impact))
    f2_flat = f2.flatten()
    frames.append({'flat': dict(f2_flat), 'status': {
        k: ('primary' if k in primary_keys else 'first' if k in first_order_keys else 'unchanged')
        for k in f2_flat
    }})

    # Frame 3 — second-order cascade
    f3 = copy.deepcopy(f2)
    second_order_keys = set()
    for src_path, src_mag in queue_next:
        if src_path not in graph.edges:
            continue
        for target, weight in graph.edges[src_path]:
            impact = src_mag * weight * dampening
            if abs(impact) >= 0.05:
                second_order_keys.add(target)
                dom_name, sub_name = target.split('.', 1)
                dom = getattr(f3, dom_name, None)
                if dom and hasattr(dom, sub_name):
                    setattr(dom, sub_name, max(0.0, min(100.0, getattr(dom, sub_name) + impact)))
    f3_flat = f3.flatten()
    frames.append({'flat': dict(f3_flat), 'status': {
        k: ('primary' if k in primary_keys else 'first' if k in first_order_keys
            else 'second' if k in second_order_keys else 'unchanged')
        for k in f3_flat
    }})

    return frames
