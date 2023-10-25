import torch.optim as optim

from src.loss.loss import BCELossWithClassWeights
from src.metrics.helper import print_metrics
from src.metrics.metrics_calculator import MetricsCalculator
from src.wiring import get_source_data, get_training_data, get_model


def train_ours_neural(object_name, query, dimension, metrics_registry):
    print(f"oursNeural {object_name} {dimension}D {query} query")

    # hyperparameters
    n_regions = 50_000
    n_samples = 1500 if dimension == 4 else 500

    # load data
    data = get_source_data(object_name=object_name, dimension=dimension)

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

    for iteration in range(total_iterations):
        features, targets = get_training_data(data=data, query=query, dimension=dimension, n_regions=n_regions,
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
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)
            print_metrics(metrics)

        if (iteration + 1) % evaluation_frequency == 0:
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)

            # if convergence to FN 0 is not stable yet and still oscillating
            # let the model continue training
            # by resetting the count
            if count != 0 and metrics["false negatives"] != 0.:
                count = 0

            # ensure that convergence to FN 0 is stable at a sufficiently large class weight
            if metrics["false negatives"] == 0.:
                count += 1

            if count == 3:
                # save final training results
                metrics_registry.metrics_registry["oursNeural"] = {
                    "class weight": class_weight,
                    "iteration": iteration+1,
                    "false negatives": metrics["false negatives"],
                    "false positives": metrics["false positives"],
                    "true values": metrics["true values"],
                    "total samples": metrics["total samples"],
                    "loss": f"{loss:.5f}"
                }

                # early stopping
                print("early stopping\n")
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

