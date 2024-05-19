import pytorch 
import numpy as np
import pandas as pd


def Linear_backward(dZ, cache):
    """Implements the linear part of the backward propagation process for a single layer
    Inputs:
        dZ – the gradient of the cost with respect to the linear output of the current layer (layer l)
        cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Output:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """
    A_prev, W, b = cache
    m = A_prev.shape[0]

    dW = torch.mm(dZ.t(), A_prev) / m
    db = torch.sum(dZ, dim=0, keepdim=True) / m
    dA_prev = torch.mm(dZ, W.t())
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies the linear_backward function.


    Input:
        dA – post activation gradient of the current layer
        cache – contains both the linear cache and the activations cache

    Output:
        dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW – Gradient of the cost with respect to W (current layer l), same shape as W
        db – Gradient of the cost with respect to b (current layer l), same shape as b
    """


