import torch.optim as optim

from src.data.data_exporter import DataExporter
from src.loss.loss import BCELossWithClassWeights
from src.metrics.metrics_calculator import MetricsCalculator
from src.metrics.metrics_registry import MetricsRegistry
from src.wiring import get_source_data, get_training_data, get_model


def train_ours_neural(object_name, query, dimension):
    # hyperparameters
    n_objects = 50_000
    n_samples = 1000 if dimension == 4 else 500

    # load data
    data = get_source_data(object_name=object_name, dimension=dimension)
    data_exporter = DataExporter(f'{object_name}_{dimension}d_{query}_query', "ours_neural")

    # initialise model
    model = get_model(query=query, dimension=dimension)

    # initialise asymmetric binary cross-entropy loss function, and optimiser
    class_weight = 1
    criterion = BCELossWithClassWeights(positive_class_weight=1, negative_class_weight=1)
    optimiser = optim.Adam(model.parameters(), lr=0.0001)

    # initialise counter and print_frequency
    weight_schedule_frequency = 250_000
    total_iterations = weight_schedule_frequency * 200  # set high iterations for early stopping to terminate training
    evaluation_frequency = weight_schedule_frequency // 5
    print_frequency = 1000  # print loss every 1k iterations

    # instantiate count for early stopping
    count = 0
    metrics_registry = MetricsRegistry()

    for iteration in range(total_iterations):
        features, targets = get_training_data(data=data, query=query, dimension=dimension, n_objects=n_objects,
                                              n_samples=n_samples)

        # forward pass
        output = model(features)

        # compute loss
        loss = criterion(output, targets)

        # zero gradients, backward pass, optimiser step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # print loss
        if (iteration + 1) % print_frequency == 0 or iteration == 0:
            print(f'Iteration: {iteration + 1}, Loss: {loss.item()}')

        if (iteration + 1) % evaluation_frequency == 0 or iteration == 0:
            out = (model(features).cpu().detach() >= 0.5).float().numpy()
            targets = targets.cpu().detach().numpy()

            MetricsCalculator.calculate(metrics_registry, prediction=out, target=targets)
            metrics = metrics_registry.get_metrics()

            for key, value in metrics.items():
                print(f"{key}: {value}")

            data_exporter.save_experiment_results(class_weight=class_weight, metrics_registry=metrics_registry,
                                                  iteration=iteration + 1, loss=loss)

        if (iteration + 1) % evaluation_frequency == 0:
            metrics = metrics_registry.get_metrics()

            # if convergence to FN 0 is not stable yet and still oscillating
            # let the model continue training
            # by resetting the count
            if count != 0 and metrics["false_negative"] != 0.:
                count = 0

            # ensure that convergence to FN 0 is stable at a sufficiently large class weight
            if metrics["false_negative"] == 0.:
                count += 1

            if count == 3:
                print("early stopping")
                break

        # schedule increases class weight by 20 every 500k iterations
        if (iteration + 1) % weight_schedule_frequency == 0 or iteration == 0:
            if iteration == 0:
                pass
            elif (iteration + 1) == weight_schedule_frequency:
                class_weight = 20
            else:
                class_weight += 20

            criterion.negative_class_weight = 1.0 / class_weight

            print("class weight", class_weight)
            print("BCE loss negative class weight", criterion.negative_class_weight)

    data_exporter.export_results()
