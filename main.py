from keras.datasets import mnist
import numpy as np
from model import L_layer_model,Predict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def process_data(x,y):
    #normalize
    x = x.reshape((x.shape[0], -1)) / 255.0

    #one hot encoding label
    one_hot = np.zeros((y.size, 10))
    # np.arange(y.size) creates an array of indices from 0 to len(y)-1, y contains the class labels
    one_hot[np.arange(y.size), y] = 1
    y=one_hot
    return x,y


#import data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# # Preprocess the data by flattening and normalizing
# X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
# X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

# # Concatenate train and test along the first axis (vertically, if you consider each image as a row in a 2D array)
# X = np.concatenate((X_train, X_test), axis=0)
# y = np.concatenate((y_train, y_test), axis=0)

# #Convert an array of labels into a one-hot representation.

# one_hot = np.zeros((y.size, 10))
# # np.arange(y.size) creates an array of indices from 0 to len(y)-1, y contains the class labels
# one_hot[np.arange(y.size), y] = 1
# y=one_hot
# m = X.shape[0]  # Total number of examples
# indices = np.arange(m)
# X_shuffled = X[indices]
# y_shuffled = y[indices]

# # Define the split size
# split = int(0.2 * m)  # 20% of the total dataset size

# # Split the data into 20/80
# X_test = X_shuffled[:split]
# y_test = y_shuffled[:split]
# X_train = X_shuffled[split:]
# y_train = y_shuffled[split:]

X_train,y_train=process_data(X_train,y_train)
X_test,y_test=process_data(X_test,y_test)
layer_dims=[X_train.shape[1],20,7,5,10]
learning_rate=0.001
num_iterations=10000
batch_size=128
L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False)