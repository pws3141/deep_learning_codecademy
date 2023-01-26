# Image classification
# cheat sheet:
# https://www.codecademy.com/learn/paths/build-deep-learning-models-with-tensorflow/tracks/dlsp-classification-track/modules/dlsp-image-classification/cheatsheet

import tensorflow as tf

# Example of large, simple model...

model = tf.keras.Sequential()

#Add an input layer that will expect grayscale input images of size 256x256:

model.add(tf.keras.Input(shape = (256, 256)))

#Use a Flatten() layer to flatten the image into a single vector:
model.add(tf.keras.layers.Flatten())

#model.add(...)

model.add(tf.keras.layers.Dense(100,activation="relu"))
model.add(tf.keras.layers.Dense(50,activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary() 

### Issues: this model is too large to optimiser over
# so, use convolutional neural netwroks instead...

####
### Convolutional Neural Networks
####

model.add(tf.keras.Input(shape=(256,256,1)))

model.add(tf.keras.layers.Conv2D(2,5,strides=3,padding="valid",activation="relu"))

#Add first max pooling layer here.
model.add(tf.keras.layers.MaxPooling2D(pool_size = (5, 5), strides = (5, 5)))

#model.add(...)

model.add(tf.keras.layers.Conv2D(4,3,strides=1,padding="valid",activation="relu"))

#Add the second max pooling layer here.
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary()
