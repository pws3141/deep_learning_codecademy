
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

# images are 256 x 256, greyscale
# create augmentation environment
training_data_generator = ImageDataGenerator(rescale = 1./256, zoom_range = 0.2,
                                             rotation_range = 15,
                                             width_shift_range = 0.08,
                                             height_shift_range = 0.08)

test_data_generator = ImageDataGenerator(rescale = 1./256, zoom_range = 0.2,
                                         rotation_range = 15,
                                         width_shift_range = 0.08,
                                         height_shift_range = 0.08)

# load the data
dir_train = "./Covid19-dataset/train/"
dir_test = "./Covid19-dataset/test/"
batch_images = 4

training_iterator = training_data_generator.flow_from_directory(dir_train,
                                                                class_mode = 'categorical',
                                                                color_mode = 'grayscale',
                                                                batch_size = batch_images)
test_iterator = test_data_generator.flow_from_directory(dir_test,
                                                        class_mode = 'categorical',
                                                        color_mode = 'grayscale',
                                                        batch_size = batch_images)

####
### Creating classification neural network
####

model = Sequential()
# input layer
# 256 x 256 images, greyscale
model.add(Input(shape = training_iterator.image_shape))

# two Conv2D layers with MaxPooling2D sandwitch
#model.add(Conv2D(5, 5, strides = 3, activation = "relu"))
#model.add(MaxPooling2D((5, 5), strides = (3, 3)))

model.add(layers.Conv2D(4, 3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3,3), strides=3))
model.add(layers.Conv2D(4, 3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3,3), strides=3))
model.add(Flatten())
#model.add(Dense(16, activation='relu'))

# output layer
# three different classifications
# using 'softmax' to obtain probs
model.add(Dense(3, activation = "softmax"))

model.summary()


# compile
model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.02),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
        )

####
### Fitting model
####

# add early stopping rule
callback = EarlyStopping(monitor = 'categorical_accuracy', patience = 25)

history = model.fit(training_iterator,
                    steps_per_epoch = training_iterator.samples / batch_images,
                    epochs = 100, 
                    validation_data = test_iterator,
                    validation_steps = test_iterator.samples / batch_images,
                    callbacks = [callback])


####
### Plot
####

# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping
fig.tight_layout()

fig.savefig('accuracy_plot.png')

####
### Confusion matrix
####

test_steps_per_epoch = np.math.ceil(test_iterator.samples / test_iterator.batch_size)
predictions = model.predict(test_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = np.math.ceil(test_iterator.samples / test_iterator.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_iterator.classes
class_labels = list(test_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)

# further information, discussion forum:
# https://discuss.codecademy.com/t/covid-19-and-pneumonia-classification/549365/4

# codecasdemy video on deep learing:
# https://www.youtube.com/watch?v=m_tRayMFhRE
