import torch
import torch.nn as nn
import math


class OursKDOP(nn.Module):
    def __init__(self, input_dim, num_planes):
        super(OursKDOP, self).__init__()
        # initialise tensors, enabling gradients for optimisation
        self.A = torch.nn.Parameter(torch.Tensor(num_planes, input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(num_planes))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, x):
        # apply the linear transformation
        Ax = torch.matmul(x, self.A.t()) + self.bias

        # apply the sigmoid activation function element-wise to Ax
        # the sigmoid function maps any real number into the range [0, 1]
        # for negative numbers, sigmoid outputs values less than 0.5
        # for positive numbers, sigmoid outputs values greater than 0.5
        sigmoid_Ax = torch.sigmoid(Ax)

        # compute the element-wise product along dimension 1 of sigmoid_Ax
        # this operation serves to approximate the condition all(x) > 0 in a differentiable manner
        # if any element in x is negative, the corresponding element in sigmoid_Ax will be less than 0.5
        # the product of such small numbers will give an even smaller number
        # conversely, if all elements in x are positive, all elements in sigmoid_Ax will be greater than 0.5
        # the product of these larger numbers will give a larger number
        approx_all = torch.prod(sigmoid_Ax, dim=1).unsqueeze(-1)

        return approx_all
