# use deep learning to predict forest cover type

#      Spruce/Fir
#      Lodgepole Pine
#      Ponderosa Pine
#      Cottonwood/Willow
#      Aspen
#      Douglas-fir
#      Krummholz

# The data is raw and has not been scaled or preprocessed. It contains binary
# columns of data for qualitative independent variables such as wilderness areas and # soil type.

# Project Objectives:

#    1. Develop one or more classifiers for this multi-class classification problem.
#    2. Use TensorFlow with Keras to build your classifier(s).
#    3. Use your knowledge of hyperparameter tuning to improve the performance of your model(s).
#    4. Test and analyze performance.
#    5. Create clean and modular code.

###
### Aim: predict 'class' of trees

# load modules

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.compose import ColumnTransformer
#from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# load data
data = pd.read_csv("cover_data.csv")

# examine data
data.shape

data.columns

data.info()

data.head()

# split into features and labels

feature_names = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Hillshade_9am"]
features = data.loc[:, feature_names]

labels = data.iloc[:, -1]

features.shape
features.info()

# split into training and test set

features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, train_size = 0.8, random_state = 22)

# standardize columns
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_train_scaled = pd.DataFrame(features_train_scaled, 
                                     columns = features_train.columns)
features_train_scaled.describe()

features_test_scaled = scaler.transform(features_test)
features_test_scaled = pd.DataFrame(features_test_scaled, 
                                     columns = features_test.columns)
features_test_scaled.describe()

# current labels from 1 - 7, shift to 0 - 6
# using 'LabelEncoder'
#le = LabelEncoder()
#labels_train = pd.DataFrame(le.fit_transform(labels_train), columns = ["class"])
#labels_test = pd.DataFrame(le.transform(labels_test), columns = ["class"])
labels_train = labels_train - 1
labels_test = labels_test - 1


#####
### Building model
#####

model = Sequential()

# input layer
model.add(InputLayer(features_train_scaled.shape[1]))

# hidden layer
model.add(Dense(64, activation = "relu"))
model.add(Dense(12, activation = "relu"))

# output layer
# seven different categories of trees
model.add(Dense(7, activation = "softmax"))

model.summary()

# compile model
# n.b. 'categorical_crossentropy' works on one-hot encoded target, while
# 'sparse_categorical_crossentropy' works on integer target. See:
# https://stackoverflow.com/questions/61550026/valueerror-shapes-none-1-and-none-3-are-incompatible
# https://www.kaggle.com/general/197993

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam",
              metrics = ["accuracy"])

# fit model to training set
# early stopping rule
callback = EarlyStopping(monitor = 'loss', patience = 5)

history = model.fit(features_train_scaled, labels_train, epochs = 20,
                    batch_size = 100, verbose = 1, callbacks = [callback])

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


### let's try a grid search
#from keras.wrappers.scikit_learn import KerasClassifier
# DEPRECATED. Use [Sci-Keras](https://github.com/adriangb/scikeras) instead.
# See https://www.adriangb.com/scikeras/stable/migration.html
import tensorflow as tf
from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import GridSearchCV

# Function to create model, required for KerasClassifier
def create_model():
	# create model
    model = Sequential()
    # input layer
    model.add(InputLayer(features_train_scaled.shape[1]))
    # hidden layer
    model.add(Dense(64, activation = "relu"))
    model.add(Dense(12, activation = "sigmoid"))
    # output layer
    # seven different categories of trees
    model.add(Dense(7, activation = "softmax"))
	# Compile model
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam",
              metrics = ["accuracy"])
	return model

# fix random seed for reproducibility
seed = 22
tf.random.set_seed(seed)

# create model
model = KerasClassifier(model = create_model, verbose = 0)

# define the grid search parameters
batch_size = [60, 80, 100]
epochs = [4, 12, 20]
param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1, cv = 3)
grid_result = grid.fit(features_train_scaled, labels_train, verbose = 1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


### random search

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_grid = {'batch_size': randint(20, 500), 'epochs': randint(5, 30)}

# 'n_iter' gives number of parameter combinations sampled
random_grid = RandomizedSearchCV(estimator = model, param_distributions = param_grid,
                          n_jobs = -1, cv = 3, n_iter = 12)
random_grid_result = random_grid.fit(features_train_scaled, labels_train, verbose = 1)

# summarize results
print("Best: %f using %s" % (random_grid_result.best_score_, random_grid_result.best_params_))
means = random_grid_result.cv_results_['mean_test_score']
stds = random_grid_result.cv_results_['std_test_score']
params = random_grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

