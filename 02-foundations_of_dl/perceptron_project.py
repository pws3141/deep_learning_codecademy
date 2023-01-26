# 2 Foundations of Deep Learning and Perceptrons
# https://www.codecademy.com/learn/paths/build-deep-learning-models-with-tensorflow

import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ]

### AND labels

labels_and = [0, 0, 0, 1]

# plot a scatter graph of the four points
x_values = [point[0] for point in data]
y_values = [point[1] for point in data]

plt.scatter(x_values, y_values, c = labels_and)

# create a perceptron to learn AND
classifier = Perceptron(max_iter = 40)
classifier.fit(data, labels_and)
print(classifier.score(data, labels_and))

# look at decision boundary for AND perceptron
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5], [0, 0.1]]))
# look at lots of points to create colour map of boundary
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = abs(distances)
distances_matrix = np.reshape(abs_distances, (100, 100))
# draw heat map of boundary
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show

### OR labels
labels_or = [0, 1, 1, 0]

classifier.fit(data, labels_or)
print(classifier.score(data, labels_or))



