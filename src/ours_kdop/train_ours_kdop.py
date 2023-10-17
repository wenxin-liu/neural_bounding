import time
from torch import optim
from src import device

from src.data.data_exporter import DataExporter
from src.loss.loss import BCELossWithClassWeights
from src.metrics.metrics_calculator import MetricsCalculator
from src.metrics.metrics_registry import MetricsRegistry
from src.ours_kdop.ours_kdop import OursKDOP
from src.wiring import get_source_data, get_training_data


def train_ours_kdop(object_name, query, dimension):
    # hyperparameters
    input_dim = dimension if query == 'point' else dimension * 2
    n_objects = 50_000
    n_samples = 1000 if dimension == 4 else 500

    # determine the number of k-DOP planes using a 4k-DOP construction
    # for example, if input_dim is 3, then 4k-DOP is a 12-DOP
    n_planes = input_dim * 4

    # load data
    data = get_source_data(object_name=object_name, dimension=dimension)
    data_exporter = DataExporter(f'{object_name}_{dimension}d_{query}_query', "ours_kdop")

    # initialise model
    model = OursKDOP(input_dim, n_planes).to(device)

    # initialise asymmetric binary cross-entropy loss function, and optimiser
    class_weight = 1
    criterion = BCELossWithClassWeights(positive_class_weight=1.0, negative_class_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # initialise counter and print_frequency
    weight_schedule_frequency = 100_000
    total_iterations = weight_schedule_frequency * 500  # set high iterations for early stopping to terminate training
    evaluation_frequency = weight_schedule_frequency // 2
    print_frequency = 1000  # print loss every 1k iterations

    # instantiate count for early stopping
    count = 0
    metrics_registry = MetricsRegistry()

    # training loop
    for iteration in range(total_iterations):
        iter_start = time.time()

        features, targets = get_training_data(data=data, query=query, dimension=dimension, n_objects=n_objects,
                                              n_samples=n_samples)

        # forward passes
        outputs = model(features)

        # compute loss
        loss = criterion(outputs, targets)

        # zero gradients, backward pass, optimiser step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging and evaluation
        if (iteration + 1) % print_frequency == 0:
            print(f"Epoch {iteration + 1}, Loss: {loss}, iteration time: {time.time() - iter_start:.5f}s")

        if (iteration + 1) % evaluation_frequency == 0 or iteration == 0:
            # evaluate metrics
            out = (model(features).cpu().detach() >= 0.5).float().numpy()
            targets = targets.cpu().detach().numpy()

            MetricsCalculator.calculate(metrics_registry, prediction=out, target=targets)
            metrics = metrics_registry.get_metrics()

            for key, value in metrics.items():
                print(f"{key}: {value}")

            data_exporter.save_experiment_results(class_weight=class_weight, metrics_registry=metrics_registry,
                                                  iteration=iteration + 1, loss=loss)

        if (iteration + 1) % weight_schedule_frequency == 0:
            metrics = metrics_registry.get_metrics()

            # early stopping logic
            # reset count if false negatives are oscillating
            if count != 0 and metrics["false_negative"] != 0.:
                count = 0

            # increment count if false negatives have stably converged to zero
            if metrics["false_negative"] == 0.:
                count += 1

            # break if convergence is stable for 6 cycles
            if count == 6:
                print("early stopping")
                break

            # adjust class weights at weight schedule frequency
            # skip if false negatives have already converged to zero
            if iteration != 0 and metrics["false_negative"] == 0.:
                pass
            # initial class weight adjustment
            elif (iteration + 1) == weight_schedule_frequency:
                class_weight = 20 if dimension == 2 else 50
            # subsequent class weight adjustments
            else:
                class_weight += 20 if dimension == 2 else 50

            # update the negative class weight in the loss criterion
            criterion.negative_class_weight = 1.0 / class_weight

            print("class weight", class_weight)
            print("BCE loss negative class weight", criterion.negative_class_weight)

    data_exporter.export_results()

