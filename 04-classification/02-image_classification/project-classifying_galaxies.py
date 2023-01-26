# Classification project: classifying images of galaxies into four different types
# “Normal”,”Ringed”,”Merger”,”Other”
# images not available...
# originally from https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app

input_data, labels = load_galaxy_data()

print(input_data.shape)
print(labels.shape)

# create training and test sets
X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size = 0.22, random_state = 222, stratify = labels)

# preprocess the input...
training_data_generator = ImageDataGenerator(rescale = 1./128)

validation_data_generator = ImageDataGenerator(rescale = 1./128)

# generate pictures using augmentation above...
training_iterator = training_data_generator.flow(X_train, y_train, batch_size = 5)
validation_iterator = validation_data_generator.flow(X_test, y_test, batch_size = 5)

## building the model
model = Sequential()
# input layer
model.add(tf.keras.Input(shape = (128, 128, 3)))

# two convolution layers
# with max pooling layers inbetween
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(tf.keras.layers.Flatten())

# hidden dense layer with 16 units
model.add(tf.keras.layers.Dense(16, activation = "relu"))

# output layer:
# Four different outputs: “Normal”,”Ringed”,”Merger”,”Other”
model.add(tf.keras.layers.Dense(4, activation = "softmax"))

model.summary() 

# compile model
adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_metric = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy()
auc = tf.keras.metrics.AUC()

model.compile(optimizer = adam, loss = loss_metric, metrics = [acc, auc])

# train the model

model.fit(
  training_iterator,
  steps_per_epoch = training_iterator.__len__() / 5,
  epochs = 12,
  validation_data = validation_iterator,
  validation_steps = validation_iterator.__len__() / 5
)

from visualize import visualize_activations
visualize_activations(model,validation_iterator)
