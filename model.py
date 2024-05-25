import numpy as np  
import pandas as pd

np.random.seed(5)

#helper functions (non-required)
def split(X,y):
    m = X.shape[0]  # Total number of examples
    indices = np.arange(m)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Define the split size
    split = int(0.2 * m)  # 20% of the total dataset size

    # Split the data into 20/80
    X_val = X_shuffled[:split]
    y_val = y_shuffled[:split]
    X_train = X_shuffled[split:]
    y_train = y_shuffled[split:]
    return X_train, y_train,X_val, y_val


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

def compute_cost_validation(AL, Y):
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
    Y=Y.T
    m = Y.shape[1]  # Number of examples
    # cross entropy loss
    cost = -(1 / m) * np.sum(Y * np.log(AL + 1e-8))  # Added a small epsilon (1e-8) to prevent log(0)
    return cost



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
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db  


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies the linear_backward function.


    Input:
        dA – post activation gradient of the current layer
        cache – contains both the linear cache and the activations cache
        activation (str): The activation function used ('relu' or 'softmax')
    Output:
        dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW – Gradient of the cost with respect to W (current layer l), same shape as W
        db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    A_prev, W, b = linear_cache
    Z = activation_cache
    if activation == "relu":
        dZ = relu_backward(dA,Z)
    elif activation == "softmax":
        dZ = softmax_backward(dA,Z)
        

    dA_prev, dW, db = Linear_backward(dZ, (A_prev, W, b))
    return dA_prev, dW, db


def relu_backward (dA, activation_cache):
    """    
    Implements backward propagation for a ReLU unit

    Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation):
    Output:
        dZ – gradient of the cost with respect to Z

    """
    Z = activation_cache
    # Create a mask that is 1 where Z > 0 and 0 elsewhere
    dZ = (Z > 0).astype(float) * dA  # Element-wise multiplication of the mask with dA

    return dZ


def softmax_backward (dA, activation_cache):
    """
    Implements backward propagation for a softmax unit

    Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation)
    output:
        dZ – gradient of the cost with respect to Z

    """

    # Since dA is assumed to be softmax(Z) - Y, directly return it as the gradient w.r.t Z

    return dA


def	L_model_backward(AL, Y, caches):
    """    Implement the backward propagation process for the entire network.

    Input
        AL - the probabilities vector, the output of the forward propagation (L_model_forward)
        Y - the true labels vector (the "ground truth" - true classifications)
        Caches - list of caches containing for each layer: a) the linear cache; b) the activation cache
    Output:
        Grads - a dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
    """
    Grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    
    # Initialize the backpropagation
    # Derivative of cross-entropy loss with respect to AL if using softmax
    dAL = AL - Y    
    # Last layer (softmax and cross-entropy loss)
    current_cache = caches[-1]
    Grads["dA" + str(L)], Grads["dW" + str(L)], Grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,activation='softmax')
    
    # Loop over the layers backward
    for l in reversed(range(L-1)):
        # lth layer: Relu backpropagation
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(Grads["dA" + str(l + 2)], current_cache,activation='relu')
        Grads["dA" + str(l + 1)] = dA_prev_temp
        Grads["dW" + str(l + 1)] = dW_temp
        Grads["db" + str(l + 1)] = db_temp

    return Grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.
    
    Parameters:
    parameters (dict): A dictionary containing the parameters of the DNN architecture.
    grads (dict): A dictionary containing the gradients, generated by L_model_backward.
    learning_rate (float): The learning rate used to update the parameters (alpha).
    
    Returns:
    parameters (dict): The updated values of the parameters object provided as input.
    """
    L = len(parameters) // 2  # Number of layers in the neural network
    
    # Update each parameter
    for l in range(0, L):
        parameters[f"W{l+1}"] = parameters[f"W{l+1}"]-learning_rate * grads[f"dW{l+1}"]
        parameters[f"b{l+1}"] = parameters[f"b{l+1}"]-learning_rate * grads[f"db{l+1}"]
    
    return parameters



def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX.
    
    Parameters:
    X -- data, numpy array of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 through num_classes-1), numpy array of shape (num_classes, number of examples)
    layer_dims -- list containing the input size and each layer size, of length (number of layers + 1)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    batch_size -- number of examples in a single training batch
    use_batchnorm -- if set to True, apply batch normalization after each activation
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- list of costs every 100 steps
    """
    np.random.seed(1)
    X, Y,X_val, Y_val=split(X,Y)
    costs = []  # keep track of cost
    val_costs = []
    best_val_cost=float('inf')
    m = X.shape[1]  # number of examples
    parameters = initialize_parameters(layer_dims)

    # Loop (gradient descent)
    for i in range(num_iterations):
        
        for j in range(0, m, batch_size):
            # Get the next batch
            begin = j
            end = min(j + batch_size, m)
            X_batch = X[ begin:end,:]
            Y_batch = Y[begin:end].T
            
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm)
            
            # Compute cost.
            cost = compute_cost(AL, Y_batch)
            
            # Backward propagation.
            grads = L_model_backward(AL, Y_batch, caches)
            
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training iterations
        if i % 100 == 0:
            AL_val, _ = L_model_forward(X_val, parameters, use_batchnorm)
            val_cost = compute_cost_validation(AL_val, Y_val)
            val_costs.append(val_cost)
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost:.6f}, Validation Cost: {val_cost:.6f}")

            # Early stopping
            if val_cost > best_val_cost:
                print("Stopping early due to increase in validation cost.")
                break
            best_val_cost = val_cost
            
    return parameters, costs


def Predict(X, Y, parameters):
    """
    Predicts the results using a trained neural network and calculates the accuracy.

    Arguments:
    X -- input data, numpy array of shape (input size, number of examples)
    Y -- true "label" vectors, numpy array of shape (num_classes, number of examples)
    parameters -- dictionary containing parameters of the DNN architecture

    Returns:
    accuracy -- the percentage of samples for which the correct label receives the highest confidence score
    """
    # Forward propagate through the network
    AL, _ = L_model_forward(X, parameters, use_batchnorm=False)  # use_batchnorm depends on your model's training
    
    # Apply softmax to the output layer's linear activations
    predictions = softmax(AL)[0]
    
    # Determine predicted labels
    predicted_labels = np.argmax(predictions, axis=0)
    true_labels = np.argmax(Y.T, axis=0)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    
    return accuracy * 100  # Convert proportion to percentage
