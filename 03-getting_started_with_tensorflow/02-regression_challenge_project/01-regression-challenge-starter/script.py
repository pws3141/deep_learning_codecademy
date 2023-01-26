import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
#from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# load data
dataset = pd.read_csv("admissions_data.csv")
# examine data
dataset.shape
dataset.head()
dataset.columns
dataset.describe()

# remove 'Serial No.' column
dataset = dataset.drop(labels = "Serial No.", axis = 1)
dataset.shape
dataset.head()


# Aim: find chance of admittion
# regression problem

# split into labels and features
num_features = dataset.shape[1] - 1 # 7 features
features = dataset.iloc[:, range(num_features)]
features.shape
features.head()

labels = dataset.iloc[:, -1]
labels.shape
labels.head()

# remove categorical data from features
# using one-hot encoding
# 'University Rating' and 'Research'
features = pd.get_dummies(features)
features.shape
features.head()

# create training and test set
(features_train, features_test,
 labels_train, labels_test) = train_test_split(features, labels,
                                               train_size = 0.7,
                                               random_state = 1)

# standardised the data 
# using training set for parameters
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_train_scaled = pd.DataFrame(features_train_scaled,
                                     columns = [features_train.columns])
features_train_scaled.describe() 
features_train_scaled.head() 

features_test_scaled = scaler.transform(features_test)
features_train_scaled = pd.DataFrame(features_test_scaled,
                                     columns = [features_test.columns])
features_train_scaled.describe() 
features_train_scaled.head()

### neural network / regression problem
# create keras Sequential model
# tensorflow.keras.models.Sequential

model = Sequential()
input = InputLayer(input_shape = features.shape[1])
model.add(input)
# one output for regression model
model.add(Dense(1))
# model summary
model.summary()

# using Adam opt
opt = Adam(learning_rate=0.01)
model.compile(loss='mse',  metrics=['mae'], optimizer=opt)

# train model
model.fit(features_train, labels_train, epochs = 40, batch_size = 1,
             verbose = 1)

#evaluate the model on the test data
val_mse, val_mae = model.evaluate(features_test, labels_test, verbose = 1)

print("MAE: ", val_mae)

#### Model Tuning
print("tuning Sequential model...")



### Tuning parameters individually
print("\ttuning model parameters individually...")
# create function that builds model and plots loss

def fit_model_learning(f_train, l_train, learning_rate, num_epochs, bs):
    #build the model
    model = Sequential()
    input = InputLayer(input_shape = f_train.shape[1])
    model.add(input)
    # one output for regression model
    model.add(Dense(1))
    opt = Adam(learning_rate = learning_rate)
    model.compile(loss='mse',  metrics=['mae'], optimizer=opt)

    #train the model on the training data
    history = model.fit(f_train, l_train, epochs = num_epochs, batch_size = bs,
                        verbose = 0, validation_split = 0.2)
    # plot learning curves
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.ylim(0, 10)
    plt.title('lrate=' + str(learning_rate))
    plt.legend(loc="upper right")

## learning rates{{{
print("\t\ttuning learning rate")

#make a list of learning rates to try out
learning_rates = [1, 1E-1, 1E-2]
#fixed number of epochs
num_epochs = 100
#fixed number of batches
batch_size = 10 

#%matplotlib

for i in range(len(learning_rates)):
  plot_no = 220 + (i+1)
  plt.subplot(plot_no)
  fit_model_learning(features_train, labels_train, learning_rates[i], num_epochs, batch_size)

plt.tight_layout()
#plt.show()
plt.savefig('plots/tuning-learning_rate.png')
plt.clf()
print("\t\t\tplotted MSE loss for learning rates: ", learning_rates)# }}}

## batch size
print("\t\ttuning batch size")

def fit_model_batch(f_train, l_train, learning_rate, num_epochs, batch_size, ax):
    #build the model
    model = Sequential()
    input = InputLayer(input_shape = f_train.shape[1])
    model.add(input)
    # one output for regression model
    model.add(Dense(1))
    opt = Adam(learning_rate = learning_rate)
    model.compile(loss='mse',  metrics=['mae'], optimizer=opt)

    #train the model on the training data
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size
                        = batch_size, verbose=1, validation_split = 0.3)
    # plot learning curves
    ax.plot(history.history['mae'], label='train')
    ax.plot(history.history['val_mae'], label='validation')
    ax.set_title('batch = ' + str(batch_size), fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_xlabel('# epochs')
    ax.set_ylabel('mae')
    ax.legend()

#fixed learning rate
learning_rate = 0.1
#fixed number of epochs
num_epochs = 100
#we choose a number of batch sizes to try out
batches = [4, 32, 64]
print("Learning rate fixed to:", learning_rate)

#plotting code
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.7, 'wspace': 0.4}) #preparing axes for plotting
axes = [ax1, ax2, ax3]

#iterate through all the batch values
for i in range(len(batches)):
  fit_model_batch(features_train, labels_train, learning_rate, num_epochs, batches[i], axes[i])

plt.savefig('plots/tuning-batch_size.png')
plt.clf()
print("\t\t\tplotted MSE loss for batch sizes: ", batches)# }}}

### TODO: 
## manual tuning
# epochs / early stopping
# Insert Layer: varying nodes
## automatic tuning
# grid search / random search
# regulisation / drop out
# baseline

####
### Predictive power of model
####

# create model
# TODO: update model to 'best' found using tuning
model = Sequential()
input = InputLayer(input_shape = features.shape[1])
model.add(input)
# one output for regression model
model.add(Dense(1))
# model summary
model.summary()
# using Adam opt
opt = Adam(learning_rate=0.01)
model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
# train model
model.fit(features_train, labels_train, epochs = 40, batch_size = 1,
             verbose = 0)
#evaluate the model on the test data
val_mse, val_mae = model.evaluate(features_test, labels_test, verbose = 0)

# predict test set
labels_predict = model.predict(features_test)
labels_predict_r2 = r2_score(labels_predict, labels_test)
print(labels_predict_r2)


