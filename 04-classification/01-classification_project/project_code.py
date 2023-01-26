# dataset from Kaggle:
# https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

from collections import Counter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical

# read and examine data
data = pd.read_csv("heart_failure.csv")
data.info()
print("Classes and number of values in the data:")
Counter(data["death_event"])

#extract features and labels

features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'high_blood_pressure', 'platelets',
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
x = data.loc[:, features]
y = data.loc[:, "death_event"]

# convert catagorical features using one-hot encoding
x = pd.get_dummies(x)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    random_state = 1)

# scale the numeric features using 'StandardScaler'
features_numerical = x.select_dtypes(include = ["float64", "int64"]).columns
ct = ColumnTransformer([("standardize", StandardScaler(), features_numerical)],
                       remainder = "passthrough") 

ct.fit_transform(X_train)
ct.transform(X_test)

# need to change labels to numerical values
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
# one-hot encording: 2d array
y_train = to_categorical(y_train, dtype = "int64")
y_test = to_categorical(y_test, dtype = "int64")


####
### Building model
####

model = Sequential()
# add input layer
model.add(InputLayer(x.shape[1]))
# add hidden layer
model.add(Dense(12, activation = "relu"))
# add output layer
model.add(Dense(2, activation = "softmax"))

# compile model
model.compile(loss = "categorical_crossentropy", optimizer = "adam",
              metrics = ["accuracy"])

# fit model to training set
model.fit(X_train, y_train, epochs = 100, batch_size = 16, verbose = 1)

# evaluate the model
loss, acc = model.evaluate(X_test, y_test)

# predict the X_test labels
y_estimate = model.predict(X_test)
# using 'np.argmax', assign probs to predicted classes
y_estimate = np.argmax(y_estimate, axis = 1)
# and to the true labels
y_true = np.argmax(y_test, axis = 1)

# print out additinal metrics
print(classification_report(y_true, y_estimate))
