import numpy as np  
import pandas as pd

np.random.seed(5)



def initialize_parameters(layer_dims):
    """
    this function initializes the parameters of the neural network
    Arguments:
    layer_dims -- numpy array containing the dimensions of the layers of the network    
    Returns:
    parameters -- python dictionary containing the initialized parameters W and b
    for each layer of the network
    """
    parameters = {}

    for l in range(1,len(layer_dims)):
 
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) *np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters



def linear_forward(A, W, b):
    """
    this function calculates the linear part of the layer's forward propagation

    Arguments:
    A -- tha activation of the previous layer
    W -- the weights matrix of the current layer shape (size of current layer, size of previous layer)
    b -- the bias vector of the current layer shape (size of current layer, 1)
    
    Returns:
    Z -- the linar component of the activation function

    linaer_cache -- a dictionary containing A, W, b to be used in backpropagation

    
    """
    zl = np.dot(W, A) + b 
    linear_cache = (A, W, b)
    return zl, linear_cache


def softmax(Z):
    """
    this function calculates the softmax activation function of the layer(Z is big Z
    which contains all the small z's of the layer)

    Arguments:
    Z -- the linear component of the activation function
    
    Returns:
    A -- the activation of the layer
    activation_cache --  returns Z to be used in backpropagation

    """
    
    # Subtract the maximum value in each column for numerical stability
    Z_shifted = Z - np.max(Z, axis=0)
    
    # Calculate exp safely
    exp_Z = np.exp(Z_shifted)
    
    # Sum across rows to normalize, avoiding adding a small constant because the issue is handled
    sum_exp_Z = np.sum(exp_Z, axis=0)
    
    # Compute the softmax activation
    A = exp_Z / sum_exp_Z
    
    # Store Z in the cache to use during the backward pass
    activation_cache = Z
    
    return A, activation_cache


def relu(Z):
    """
    this function calculates the relu activation function of the layer(Z is big Z
    which contains all the small z's of the layer)

    Arguments:
    Z -- the linear component of the activation function
    
    Returns:
    A -- the activation of the layer
    activation_cache --  returns Z to be used in backpropagation

    """
    A = np.maximum(0,Z)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    this function calculates the forward propagation of the layer

    Arguments:
    A_prev -- the activation of the previous layer
    W -- the weights matrix of the current layer shape (size of current layer, size of previous layer)
    B -- the bias vector of the current layer shape (size of current layer, 1)
    activation -- the activation function to be used in the layer
    
    Returns:
    A -- the activation of the current layer
    cache --  a joint dictinary containing the linear_cache and activation_cache 


    """
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, use_batchnorm):
    """
    this function calculates the forward propagation of the neural network 
    using the linear -> relu * (L-1) -> linear -> softmax architecture computation  

    Arguments:
    X -- the input data of shape (input size, number of examples)
    parameters -- the parameters of the neural network W and b for each layer
    use_batchnorm -- a boolean variable to determine whether to use batch normalization or not
    
    Returns:
    AL -- the activation of the output layer
    caches --  a list of all the caches of the layers

    """
    caches = []
    A = X.T
    for l in range(0, int(len(parameters)/2)-1):

        if use_batchnorm:
            A = apply_batchnorm(A)
        
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l+1)], parameters['b' + str(l+1)], "relu")
        caches.append(cache)

    l=int(len(parameters)/2)
    # output layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(l)], parameters['b' + str(l)], "softmax")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    """
    this function calculates the cross entropy cost of the neural network

    using categorical cross entropy loss function
    
    Arguments:
    AL -- the activation of the output layer
    Y -- the true labels of the data
    
    Returns:
    cost -- the cross entropy cost of the neural network

    """
    
    # soft max for AL
    m = Y.shape[1]  # Number of examples
    # cross entropy loss
    cost = -(1 / m) * np.sum(Y * np.log(AL + 1e-8))  # Added a small epsilon (1e-8) to prevent log(0)
    return cost



def apply_batchnorm(A,gamma=1,beta=0,epsilon=1e-2):
    """
    this function applies batch normalization on the activation values of the layer

    Arguments:
    A -- the activation of the layer

    Returns:
    NA -- the normalized activation of the layer

    """
    # mean
    mu = np.mean(A, axis=1)
    # variance
    var = np.var(A, axis=1)

    zi = (A - mu) / np.sqrt(var + epsilon)

    NA = gamma * zi + beta
    return NA



