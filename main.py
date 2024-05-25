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
X_train,y_train=process_data(X_train,y_train)
X_test,y_test=process_data(X_test,y_test)

#initilize model
layer_dims=[X_train.shape[1],20,7,5,10]
learning_rate=0.001
num_iterations=4000
batch_size=128
params,costs=L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False)
Predict(X_test,y_test,params)