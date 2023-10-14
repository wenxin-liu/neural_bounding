from src import device
from src.indicator.indicator import indicator
from src.indicator.region_samplers import sample_points
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

from src.loss.loss import BCELossWithClassWeights
from src.metrics.metrics_calculator import MetricsCalculator
from src.metrics.metrics_registry import MetricsRegistry
from src.ours_neural.nn_model_2d import OursNeural2D


def train_ours_neural_2d_point():
    # hyperparameters
    input_dim = 2
    n_sample = 50000

    # load data
    data = torch.tensor(plt.imread('bunny_32.png')[:, :, 3] >= 0.5).float().to(device)

    # initialise model, asymmetric binary cross-entropy loss function, and optimiser
    model = OursNeural2D(input_dim).to(device)
    class_weight = 1
    criterion = BCELossWithClassWeights(positive_class_weight=1, negative_class_weight=1)
    optimiser = optim.Adam(model.parameters(), lr=0.0001)

    # initialise counter and print_frequency
    weight_schedule_frequency = 500_000
    total_iterations = weight_schedule_frequency * 30  # set large total number of iterations for early stopping to terminate training
    evaluation_frequency = weight_schedule_frequency // 10
    print_frequency = 1000  # print loss every 1k iterations
    metrics_frequency = 10000  # show metrics every 10k iterations

    # instantiate count for early stopping
    count = 0
    metrics_registry = MetricsRegistry()

    for iteration in range(total_iterations):
        # generate n_sample of random sample points
        x_sample = sample_points(n_sample, input_dim).to(device)

        # generate the corresponding targets using indicator function
        y_sample = indicator(x_sample, data).to(device)

        # forward pass
        output = model(x_sample)

        # compute loss
        loss = criterion(output, y_sample)

        # zero gradients, backward pass, optimiser step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # increment counter

        # print loss
        if (iteration + 1) % print_frequency == 0 or iteration == 0:
            print(f'Iteration: {iteration + 1}, Loss: {loss.item()}')

        if (iteration + 1) % metrics_frequency == 0 or iteration == 0:
            out = (model(x_sample).cpu().detach() >= 0.5).float().numpy()
            targets = y_sample.cpu().detach().numpy()

            MetricsCalculator.calculate(metrics_registry, prediction=out, target=targets)
            metrics = metrics_registry.get_metrics()

            for key, value in metrics.items():
                print(f"{key}: {value}")

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

        # schedule increases class weight by 10 every 3M iterations
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
