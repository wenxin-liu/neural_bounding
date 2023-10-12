import numpy as np


class MetricsCalculator:
    @staticmethod
    def calculate(metrics_registry, target, prediction):
        # register metrics for evaluating model binary classification performance
        metrics_registry.register_counter_metric("false_negative")
        metrics_registry.register_counter_metric("false_positive")
        metrics_registry.register_counter_metric("true_value")
        metrics_registry.register_counter_metric("total_samples")

        difference = target - prediction
        unique, counts = np.unique(difference, return_counts=True)

        result = dict(zip(unique, counts))

        metrics_registry.add("false_negative", result.get(1.0, 0.0))
        metrics_registry.add("false_positive", result.get(-1.0, 0.0))
        metrics_registry.add("true_value", result.get(0.0, 0.0))
        metrics_registry.add("total_samples", difference.size)
