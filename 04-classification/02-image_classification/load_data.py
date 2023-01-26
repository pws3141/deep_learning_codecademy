# Image classification
# cheatsheet:
# https://www.codecademy.com/learn/paths/build-deep-learning-models-with-tensorflow/tracks/dlsp-classification-track/modules/dlsp-image-classification/cheatsheet
# data from: 
# https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

from preprocess import training_data_generator

DIRECTORY = "data/train"
#data/train is a folder that contains two subfolders:
    #NORMAL : Chest x-rays of patients without pneumonia.
    #PNEUMONIA : Chest x-rays of patients with pneumonia.
#'flow_from_directory' will automatically label the images according to their subfolder.

CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32

#Creates a DirectoryIterator object using the above parameters:

training_iterator = training_data_generator.flow_from_directory(DIRECTORY,class_mode=CLASS_MODE,color_mode=COLOR_MODE,target_size=TARGET_SIZE,batch_size=BATCH_SIZE)

sample_batch_input, sample_batch_labels = training_iterator.next()

print(sample_batch_input.shape, sample_batch_labels.shape)
