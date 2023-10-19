def print_metrics(metrics_registry):
    metrics = metrics_registry.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("")
