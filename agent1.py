"""

part 3 , using the nural network to predict the test data

some terminology:

A sample is a single row of data.

It contains inputs that are fed into the algorithm and an output that is used to 

compare to the prediction and calculate an error.

A training dataset is comprised of many rows of data, e.g. 

many samples. A sample may also be called an instance, an observation, an input vector, 

or a feature vector.



"""

#%%-----------------------------------------------------------------------
from q1 import *
from code_1 import *  
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder





np.random.seed(5)


def L_layer_model(X, Y, layers_dims, learning_rate=0.3, num_iterations=3000, batch_size=32):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function, 
    and the final layer will apply the softmax activation function. The size of the output layer 
    should be equal to the number of labels in the data. Please select a batch size that 
    enables your code to run well (i.e. no memory overflows while still running relatively fast).

    Arguments:
    X -- data, numpy array of shape (height*width, number of examples)
    Y -- true "label" vector (number of classes, number of examples)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    layers_dims -- list containing the dimensions of each layer in the network including the input layer
    batch_size -- the number of examples in a single training batch.


    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict. 
    from the update_parameters function
    costs -- the values of the cost function (calculated by the cross entropy function) over time, 
    from the compute_cost function. one value every 100 iterations.
    
    
    """
 
    costs = []                       
    m = X.shape[1]                    
    parameters = initialize_parameters(layers_dims)
    num_batches = m // batch_size
    for i in range(0, num_iterations):
        
        for j in range(num_batches):

            start = j * batch_size
            end = start + batch_size
            X_batch = X[:, start:end]
            Y_batch = Y[:, start:end]


            al, caches = L_model_forward(X_batch, parameters)
     

            cost = compute_cost(al, Y_batch)
            grads = L_model_backward(al, Y_batch, caches)

 
            parameters = update_parameters(parameters, grads, learning_rate)

            # print(parameters)

        costs.append(cost)
        print("Cost after iteration %i: %f" % (i, cost))
        # see if there is no improvement in the cost
        # if len(costs) > 2 and costs[-2] - costs[-1] < 1e-5:
        #     break

    return parameters, costs



def predict(X, Y, parameters):
    """
    this function computes the accuracy of the model based on the test data
    it uses softmax to normalize the output of the model to a probability distribution
    

    Arguments:
    X -- data, numpy array of shape (height*width, number of examples)
    Y -- true "label" vector (number of classes, number of examples)
    parameters -- parameters learnt by the model. They can then be used to predict.

    Returns:
    accuracy -- the accuracy of the model as a percentage of correct predictions
    
    """


    al, caches = L_model_forward(X, parameters)
    al = softmax(al)
    predictions = np.argmax(al, axis=0)
    Y = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == Y)
    print("Accuracy: %f" % accuracy)
    return accuracy


# %%-------------------------MNIST dataset-------------------------------------
# Load the MNIST dataset

X, y = fetch_openml('mnist_784', version=1, return_X_y=True,parser='auto')

# Normalize the data
X = X / 255.0
# turn into numpy arrays
X = np.array(X)

# One hot encode the labels to get a vector of 10 classes
y = np.array(y, dtype=int)
Y = y.reshape(1, -1)
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y.T)


# Split the data into a training and test set 20% test size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Transpose the data to have the samples as columns
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T



# %%-------------------------Train the model-------------------------------------

# Define the layers dims
layers_dims = [X_train.shape[0], 20, 7, 5, 10]

# Train the model
parameters, costs = L_layer_model(X_train, Y_train, layers_dims, learning_rate=9, num_iterations=3000)

# %%-------------------------Predict the test data-------------------------------------

# Predict the test data
accuracy = predict(X_test, Y_test, parameters)
print(accuracy)

a = np.array([[0.001,.00002,.0003],[.0000000004,.0,.0000000006],[.0000000007,.0000000008,.0000000009]])
w = np.array([[0.01,.0000000002,.0000000003],[.0,.0000000005,.0000000006]])
b = np.array([1,2])

np.dot(w,a.T) + b