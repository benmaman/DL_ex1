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
from matplotlib import pyplot as plt
from q1 import *
from code_1 import *  
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import mnist





def convert_prob_into_class(AL):
    pred = np.copy(AL)
    pred[AL > 0.5]  = 1
    pred[AL <= 0.5] = 0
    return pred

def get_accuracy(AL, Y):
    pred = convert_prob_into_class(AL)
    return (pred == Y).all(axis=0).mean()




def L_layer_model(X, Y, layers_dims, learning_rate=0.009, num_iterations=3000, batch_size=32):
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
    acu = []                
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
            accuracy =get_accuracy(al,Y_batch)
            cost = compute_cost(al, Y_batch)
            grads = L_model_backward(al, Y_batch, caches)
            
 
            parameters = update_parameters(parameters, grads, learning_rate)

            # print(parameters)

        costs.append(cost)
        acu.append(accuracy)
        if(i % 100 ==0):
            print('i='+str(i)+' cost = ' + str(cost))
            print('i='+str(i)+' accuracy = '+str(accuracy))
    return parameters, costs, acu



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
# # Load the MNIST dataset

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
y_train = Y_train.T
y_test = Y_test.T



# %%-------------------------Train the model-------------------------------------

# Define the layers dims
layers_dims = [X_train.shape[0], 20, 7, 5, 10]

# Train the model
parameters, costs , accuracies = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.009, num_iterations=3000)

# %%-------------------------Predict the test data-------------------------------------

# plot the costs and accuracies
plt.figure(figsize=(10, 6)) 
plt.plot(costs)
plt.title('Cost vs Iteration without Batch Normalization',size=15)
plt.xlabel('Iteration Number',size=15)
plt.ylabel('Cost Value',size=15)
plt.xticks(size=14)
plt.yticks(size=14)
plt.savefig('plots/plot_cost_vs_iteration.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(accuracies)
plt.title('Accuracy vs Iteration without Batch Normalization',size=15)
plt.xlabel('Iteration Number',size=15)
plt.ylabel('Accuracy Value',size=15)
plt.xticks(size=14)
plt.yticks(size=14)
plt.savefig('plots/plot_accuracy_vs_iteration.png', dpi=300)
plt.show()






# Predict the test data
accuracy = predict(X_test, y_test, parameters)
print(accuracy)

