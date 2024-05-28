
# %% Importing necessary libraries
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import L_layer_model,Predict
from l2_reg_model import l2_reg_L_layer_model
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
learning_rate=0.009
num_iterations=10000
batch_size=512



# %%------------ no batch normalization
params,costs=L_layer_model(X_train, y_train, layer_dims, learning_rate, 
                           num_iterations, batch_size, use_batchnorm=False)
Predict(X_test,y_test,params)

df=pd.DataFrame({'cost':costs})
df["iteration"]=df.index*100
df.to_csv('output/result_q4.csv')
# Create scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='iteration', y='cost', color='blue')  # Scatter plot of points

# Connect the points with a line
plt.plot(df['iteration'], df['cost'], 'r-')  # 'r-' specifies a red solid line

plt.title('Cost vs Iteration without Batch Normalization',size=15)
# Set labels
plt.xlabel('Iteration Number',size=15)
plt.ylabel('Cost Value',size=15)
plt.xticks(size=14)
plt.yticks(size=14)

plt.savefig('plots/plot_cost_vs_iteration.png', dpi=300)  # Saves the figure to a PNG file with 300 DPI

plt.show()






# %%------------ lbatch normalization


# params,costs=L_layer_model(X_train, y_train, layer_dims, learning_rate,
#                             num_iterations, batch_size, use_batchnorm=True)

# Predict(X_test,y_test,params,user_batchnorm=True)

# df=pd.DataFrame({'cost':costs})
# df["iteration"]=df.index*100
# df.to_csv('output/result_q5_batch_norm.csv')
# # Create scatter plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='iteration', y='cost', color='blue')  # Scatter plot of points

# # Connect the points with a line
# plt.plot(df['iteration'], df['cost'], 'r-')  # 'r-' specifies a red solid line

# plt.title('Cost vs Iteration with Batch Normalization',size=15)
# # Set labels
# plt.xlabel('Iteration Number',size=15)
# plt.ylabel('Cost Value',size=15)
# plt.xticks(size=14)
# plt.yticks(size=14)

# plt.savefig('plots/plot_cost_vs_iteration_5.png', dpi=300)  # Saves the figure to a PNG file with 300 DPI

# plt.show()



# %%------------ l2 norm regularization


params_l2,costs=l2_reg_L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False)
Predict(X_test,y_test,params_l2)

df=pd.DataFrame({'cost':costs})
df["iteration"]=df.index*100
df.to_csv('output/result_q6_l2_reg.csv')
# Create scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='iteration', y='cost', color='blue')  # Scatter plot of points

# Connect the points with a line
plt.plot(df['iteration'], df['cost'], 'r-')  # 'r-' specifies a red solid line

plt.title('Cost vs Iteration with L2 Regularization',size=15)
# Set labels
plt.xlabel('Iteration Number',size=15)
plt.ylabel('Cost Value',size=15)
plt.xticks(size=14)
plt.yticks(size=14)

plt.savefig('plots/plot_cost_vs_iteration_6.png', dpi=300)  # Saves the figure to a PNG file with 300 DPI

plt.show()
# %%


# ----------------------------plot whights ----------------------------


import matplotlib.pyplot as plt
import numpy as np


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


for w in params.keys():
    if w[0]=='W':
        plt.figure()
        plt.title(w)
        hinton(params[w])
        plt.savefig(f'plots/{w}.png', dpi=300)  # Saves the figure to a PNG file with 300 DPI
        plt.show()


# plot histogra for all the weights

for w in params.keys():
    if w[0]=='W':
        plt.figure()
        plt.title(w)
        plt.hist(params[w].flatten(), bins=50)
        plt.savefig(f'plots/{w}_hist.png', dpi=300)  # Saves the figure to a PNG file with 300 DPI
        plt.show()

# %%--------------------------for l2 norm regularization

for w in params_l2.keys():
    if w[0]=='W':
        plt.figure()
        plt.title(w)
        hinton(params_l2[w])
        plt.savefig(f'plots/{w}.png', dpi=300)  # Saves the figure to a PNG file with 300 DPI
        plt.show()

# plot histogra for all the weights

for w in params_l2.keys():
    if w[0]=='W':
        plt.figure()
        plt.title(w)
        plt.hist(params_l2[w].flatten(), bins=50)
        plt.savefig(f'plots/{w}_hist.png', dpi=300)  # Saves the figure to a PNG file with 300 DPI
        plt.show()
