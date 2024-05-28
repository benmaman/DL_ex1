
"""


here is the model with the l2 norm regularization and the batch normalization





"""
from numpy.linalg import matrix_power
from model import *



def l2_norm_cost(AL, Y, parameters, epsi=0.01):
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
    cost = -(1 / m) * np.sum(Y * np.log(AL + 1e-8)) 
    #     # compute the regularization penalty
    L = len(parameters) // 2  # number of layers in the neural network
    l2_cost = 0
    for l in range(1, L + 1):
        l2_cost += np.sum(np.square(parameters[f"W{l}"]))
    l2_cost *= (epsi / (2 * m))
    # add the l2 regularization penalty to the cost
    cost += l2_cost

    return cost


def	l2_reg_L_model_backward(AL, Y, caches, epsi=0.01):
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
    
    current_cache = caches[-1]
    # Initialize the backpropagation
    # Derivative of cross-entropy loss with respect to AL if using softmax with l2 regularization
    big_w = current_cache[0][1]
    

    dAL = AL - Y
    # Last layer (softmax and cross-entropy loss)
    
    Grads["dA" + str(L)], Grads["dW" + str(L)], Grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,activation='softmax')
    Grads["dW" + str(L)] = Grads["dW" + str(L)] + (epsi)*big_w
    # Loop over the layers backward
    for l in reversed(range(L-1)):
        # lth layer: Relu backpropagation
        current_cache = caches[l]
        big_w = current_cache[0][1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(Grads["dA" + str(l + 2)], current_cache,activation='relu')
        Grads["dA" + str(l + 1)] = dA_prev_temp
        Grads["dW" + str(l + 1)] = dW_temp + (epsi)*big_w
        Grads["db" + str(l + 1)] = db_temp




    return Grads

def l2_reg_update_parameters(parameters, grads, learning_rate,epsi=0.01):
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
        parameters[f"W{l+1}"] = (1- learning_rate*epsi)*parameters[f"W{l+1}"]-learning_rate * grads[f"dW{l+1}"]
        parameters[f"b{l+1}"] = parameters[f"b{l+1}"]-learning_rate * grads[f"db{l+1}"]
    
    return parameters




def l2_reg_L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):
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
    np.random.seed(22)
    X, Y,X_val, Y_val=split(X,Y)
    costs = []  # keep track of cost
    val_costs = []
    best_val_cost=float('inf')
    m = X.shape[0]  # number of examples
    parameters = initialize_parameters(layer_dims)
    iteration_counter = 0  # Total number of iterations

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
            
            cost = l2_norm_cost(AL, Y_batch, parameters, epsi=0.01)
            # Backward propagation.
            grads = l2_reg_L_model_backward(AL, Y_batch, caches, epsi=0.01)
            
            # Update parameters.
            parameters = l2_reg_update_parameters(parameters, grads, learning_rate,epsi=0.01)
        
            iteration_counter += 1
        # Print the cost every 100 training iterations
            if iteration_counter % 100 == 0:
                # Training performance
                AL_train, _ = L_model_forward(X, parameters, use_batchnorm)
                train_cost =cost
                train_acc = compute_accuracy(AL_train, Y)
                
                # Validation performance
                AL_val, _ = L_model_forward(X_val, parameters, use_batchnorm)
                val_cost = compute_cost_validation(AL_val, Y_val)
                val_acc = compute_accuracy(AL_val, Y_val)

                # Append to lists
                costs.append(train_cost)

                print(f"Iter {iteration_counter} and epoch {i}: Train Cost: {train_cost:.6f}, Val Cost: {val_cost:.6f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

                # Early stopping
                if val_cost > best_val_cost:
                    print("Stopping early due to increase in validation cost.")
                    return parameters, costs
                best_val_cost = val_cost
            
    return parameters, costs

