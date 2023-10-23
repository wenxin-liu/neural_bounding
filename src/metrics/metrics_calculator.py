import numpy as np


class MetricsCalculator:
    @staticmethod
    def calculate(target, prediction):
        difference = target - prediction
        unique, counts = np.unique(difference, return_counts=True)

        result = dict(zip(unique, counts))

        return {
            "false negatives": result.get(1.0, 0.0),
            "false positives": result.get(-1.0, 0.0),
            "true values": result.get(0.0, 0.0),
            "total samples": difference.size
        }