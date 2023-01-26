### Deep Learning: Classification
## https://www.codecademy.com/learn/paths/build-deep-learning-models-with-tensorflow/tracks/dlsp-classification-track/modules/dlsp-classification/cheatsheet

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense

#your code here
train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
print(test_data.info())

#print the class distribution

print("Classes and number of values in the dataset:")
Counter(train_data["Air_Quality"])

#extract the features from the training data
# we are looking at features: ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
# 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
            'Benzene', 'Toluene', 'Xylene', 'AQI']
x_train = train_data.loc[:, features]

#extract the label column from the training data
y_train = train_data.loc[:, "Air_Quality"]

#extract test set
x_test = test_data.loc[:, features]
y_test = test_data.loc[:, "Air_Quality"]

# convert labels from categorical to numerical data
# using LabelEncoder
# not 'get_dummies' as want 1D array

#encode the labels into integers
le = LabelEncoder()
y_train  = le.fit_transform(y_train.astype(str))
y_test  = le.transform(y_test.astype(str))

#print the integer mappings
integer_mapping = {l: i for i, l in enumerate(le.classes_)}
print("The integer mapping:\n", integer_mapping)

#convert the integer encoded labels into binary vectors
# to calculate cross-entropy
# one-hot encoding vector
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = "int64")
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = "int64")

#design the model
model = Sequential()

#add the input layer
input_layer = InputLayer(input_shape = (len(features), ))
model.add(input_layer)

#add a hidden layer
hidden_layer = Dense(10, activation = "relu")
model.add(hidden_layer)

#add an output layer
output_layer = Dense(6, activation = "softmax")
model.add(output_layer)

model.summary()

### setting the optimiser and cross-entropy
# specify cross-entropy by assigning 'loss' to 'categorical_crossentropy'
# use 'acuracy' metric to evaluate model: gives perc of correct predictions
# use Adam optimisation

model.compile(loss = "categorical_crossentropy", optimizer = "adam",
              metrics = ['accuracy'])

### Train and evaluate classification model
model.fit(x_train, y_train, epochs = 20, batch_size = 4, verbose = 1)

# loss and accuracy of test set
loss, acc = model.evaluate(x_test, y_test, verbose=0)

## F1 score using 'classification_report'
# predict classes of test set
y_estimate = model.predict(x_test)
# convert one-hot encoded predictions into index of class the sample belongs to
y_estimate = np.argmax(y_estimate, axis = 1)
# convert true test labels into index of class
y_true = np.argmax(y_test, axis = 1)

print(classification_report(y_true, y_estimate))

### Different model:
## compile model with sparse categorical cross-entropy
# here, do not need one-hot encoding

x_train = train_data.loc[:, features]
#extract the label column from the training data
y_train = train_data.loc[:, "Air_Quality"]
#extract test set
x_test = test_data.loc[:, features]
y_test = test_data.loc[:, "Air_Quality"]
#encode the labels into integers
le = LabelEncoder()
y_train  = le.fit_transform(y_train.astype(str))
y_test  = le.transform(y_test.astype(str))
#design the model
model = Sequential()
input_layer = InputLayer(input_shape = (len(features), ))
model.add(input_layer)
hidden_layer = Dense(10, activation = "relu")
model.add(hidden_layer)
output_layer = Dense(6, activation = "softmax")
model.add(output_layer)

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam",
              metrics = ['accuracy'])
#train and evaluate the model
model.fit(x_train, y_train, epochs = 20, batch_size = 16, verbose = 0)

#get additional statistics
y_estimate = model.predict(x_test, verbose=1)
y_estimate = np.argmax(y_estimate, axis=1)
print(classification_report(y_test, y_estimate))

### Tuning the model
# changing number of epochs to 30
model.fit(x_train, y_train, epochs = 30, batch_size = 16, verbose = 0)
y_estimate = model.predict(x_test, verbose=1)
y_estimate = np.argmax(y_estimate, axis=1)
print(classification_report(y_test, y_estimate))
