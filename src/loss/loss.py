import torch
import torch.nn as nn


class BCELossWithClassWeights(nn.Module):
    def __init__(self, positive_class_weight=1.0, negative_class_weight=1.0):
        super(BCELossWithClassWeights, self).__init__()

        # initialise positive and negative class weights
        self.positive_class_weight = positive_class_weight
        self.negative_class_weight = negative_class_weight

    def forward(self, predictions, targets):
        epsilon = 1e-7  # small constant to avoid division by zero

        # compute the loss for the positive class
        positive_class = targets * torch.log(predictions + epsilon).clamp(min=-100.0) * self.positive_class_weight

        # compute the loss for the negative class
        negative_class = (1.0 - targets) * torch.log(1.0 - predictions + epsilon).clamp(min=-100.0) * self.negative_class_weight

        # combine and average the loss
        total_loss = -(positive_class + negative_class).mean()

        return total_loss
