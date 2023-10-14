class MetricsRegistry:
    def __init__(self):
        self.metrics_registry = {}

    def register_counter_metric(self, name):
        self.metrics_registry[name] = 0.0

    def get_metric(self, name):
        return {name: self.metrics_registry[name]}

    def get_metrics(self):
        return self.metrics_registry

    def increment_counter_metric(self, name):
        if name in self.metrics_registry.keys():
            self.metrics_registry[name] += 1.0
        else:
            raise ValueError("metric {} not registered".format(name))

    def add(self, name, value):
        if name in self.metrics_registry.keys():
            self.metrics_registry[name] += value
        else:
            raise ValueError("metric {} not registered".format(name))

    def reset_metric(self, name):
        self.metrics_registry[name] = 0.0

    def reset_metrics(self):
        for name in self.metrics_registry.keys():
            self.metrics_registry[name] = 0.0
