import time
from torch import optim
from src import device

from src.loss.loss import BCELossWithClassWeights
from src.metrics.helper import print_metrics
from src.metrics.metrics_calculator import MetricsCalculator
from src.ours_kdop.ours_kdop import OursKDOP
from src.wiring import get_source_data, get_training_data


def train_ours_kdop(object_name, query, dimension, metrics_registry):
    print(f"oursKDOP {object_name} {dimension}D {query} query")

    # hyperparameters
    input_dim = dimension if query == 'point' else dimension * 2
    n_regions = 50_000
    n_samples = 1000 if dimension == 4 else 500

    # determine the number of k-DOP planes using a 4k-DOP construction
    # for example, if input_dim is 3, then 4k-DOP is a 12-DOP
    n_planes = input_dim * 4

    # load data
    data = get_source_data(object_name=object_name, dimension=dimension)

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

    # training loop
    for iteration in range(total_iterations):
        iter_start = time.time()

        features, targets = get_training_data(data=data, query=query, dimension=dimension, n_regions=n_regions,
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
        if (iteration + 1) % print_frequency == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss}, iteration time: {time.time() - iter_start:.5f}s")

        if (iteration + 1) % evaluation_frequency == 0 or iteration == 0:
            # evaluate metrics
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)
            print_metrics(metrics)

        if (iteration + 1) % weight_schedule_frequency == 0:
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)

            # early stopping logic
            # reset count if false negatives are oscillating
            if count != 0 and metrics["false negatives"] != 0.:
                count = 0

            # increment count if false negatives have stably converged to zero
            if metrics["false negatives"] == 0.:
                count += 1

            # break if convergence is stable for 6 cycles
            if count == 6:
                # save final training results
                metrics_registry.metrics_registry["oursKDOP"] = {
                    "class weight": class_weight,
                    "iteration": iteration + 1,
                    "false negatives": metrics["false negatives"],
                    "false positives": metrics["false positives"],
                    "true values": metrics["true values"],
                    "total samples": metrics["total samples"],
                    "loss": f"{loss:.5f}"
                }

                # early stopping
                print("early stopping\n")
                break

            # adjust class weights at weight schedule frequency
            # skip if false negatives have already converged to zero
            if iteration != 0 and metrics["false negatives"] == 0.:
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
