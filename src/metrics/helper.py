def print_metrics(metrics):
    for key, value in metrics.items():
        if key == "false negatives" or key == "false positives" or key == "true values" or key == "total samples":
            print(f"{key}: {value}")
    print("")
