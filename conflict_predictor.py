"""
conflict_predictor.py — Proactive intelligence and trajectory forecasting
"""

import copy
from life_state import LifeMetrics, DependencyGraph

class ConflictPredictor:
    def __init__(self):
        self.graph = DependencyGraph()
        self.snapshots = [] # list of flattened LifeMetrics dicts
        self.MAX_HISTORY = 10
        self.INVERSE_METRICS = {
            "mental_wellbeing.stress_level", 
            "career.workload",
            "finances.debt_pressure", 
            "time.commute_burden", 
            "time.admin_overhead"
        }

    def add_snapshot(self, metrics: LifeMetrics) -> None:
        self.snapshots.append(metrics.flatten())
        if len(self.snapshots) > self.MAX_HISTORY:
            self.snapshots.pop(0)

    def compute_trajectory(self, metric_path: str) -> float:
        if len(self.snapshots) < 3:
            return 0.0
        
        # Use last 5 snapshots maximum
        n = min(5, len(self.snapshots))
        y = [s.get(metric_path, 0.0) for s in self.snapshots[-n:]]
        x = list(range(n))
        
        # Simple linear regression: slope = Cov(x, y) / Var(x)
        mean_y = sum(y) / n
        mean_x = sum(x) / n
        cov_xy = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
        var_x = sum((x_i - mean_x) ** 2 for x_i in x)
        
        if var_x == 0:
            return 0.0
        return cov_xy / var_x
        
    def predict_crisis(self, horizon_days: int = 7) -> list:
        if not self.snapshots:
            return []
            
        current = self.snapshots[-1]
        warnings = []
        
        for metric, val in current.items():
            slope = self.compute_trajectory(metric)
            if slope == 0.0:
                continue
                
            projected = val + (slope * horizon_days)
            is_inverse = metric in self.INVERSE_METRICS
            
            # Normal metric: Critical is low (<30), Warning is low (<45)
            # Inverse metric: Critical is high (>70), Warning is high (>55)
            critical_now = (val > 70) if is_inverse else (val < 30)
            warning_now = (val > 55) if is_inverse else (val < 45)
            
            critical_proj = (projected > 70) if is_inverse else (projected < 30)
            warning_proj = (projected > 55) if is_inverse else (projected < 45)
            
            worse_direction = (slope > 0) if is_inverse else (slope < 0)
            
            if worse_direction and (critical_proj or warning_proj):
                threshold = 70.0 if is_inverse else 30.0
                days_until_crit = (threshold - val) / slope if slope != 0 else float('inf')
                
                if critical_now:
                    days_until_crit = 0.0
                
                severity = 'crisis' if critical_proj else 'warning'
                direction_word = "rising" if slope > 0 else "declining"
                friendly_name = metric.split('.')[-1].replace('_', ' ')
                
                if severity == 'crisis':
                    msg = f"{friendly_name} will hit critical levels in {max(0, int(days_until_crit))} days."
                else:
                    msg = f"{friendly_name} has been {direction_word} ({slope:+.1f}/day) — warning levels likely within {horizon_days} days."
                    
                warnings.append({
                    "metric": metric,
                    "current_value": val,
                    "projected_value": projected,
                    "days_until_critical": max(0.0, days_until_crit),
                    "severity": severity,
                    "message": msg
                })
                
        # Sort by urgency (days until critical)
        warnings.sort(key=lambda x: x['days_until_critical'])
        return warnings

    def get_prediction_summary(self) -> str:
        warnings = self.predict_crisis()
        if not warnings:
            return "Your life metrics are stable. No immediate crises predicted."
            
        messages = [w['message'] for w in warnings]
        return "Based on your current trajectory: " + " ".join(messages[:3]) + ("" if len(messages) <= 3 else " (+ more warnings hidden).")

    def get_risk_score(self) -> float:
        warnings = self.predict_crisis()
        if not warnings:
            return 0.0
            
        score = 0.0
        for w in warnings:
            if w['severity'] == 'crisis':
                score += 0.3
            else:
                score += 0.1
        return min(1.0, score)

def main():
    import random
    
    predictor = ConflictPredictor()
    
    print("Simulating 5 days of accumulating stress and declining sleep...\n")
    current_state = LifeMetrics()
    
    for i in range(5):
        current_state.mental_wellbeing.stress_level += 5.0 + random.uniform(0, 2)
        current_state.physical_health.sleep_quality -= 4.0 + random.uniform(0, 2)
        current_state.time.free_hours_per_week -= 1.0 + random.uniform(0, 1)
        
        predictor.add_snapshot(current_state)
        print(f"Day {i+1}: Stress={current_state.mental_wellbeing.stress_level:.1f}, Sleep={current_state.physical_health.sleep_quality:.1f}")
        
    print("\n--- PREDICTION AFTER 5 DAYS ---")
    print(f"Risk Score: {predictor.get_risk_score():.2f}")
    print("Summary:")
    print(predictor.get_prediction_summary())

if __name__ == '__main__':
    main()
