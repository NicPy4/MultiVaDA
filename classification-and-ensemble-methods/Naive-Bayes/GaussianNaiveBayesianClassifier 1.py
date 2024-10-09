# Code based on example from https://www.geeksforgeeks.org/naive-bayes-classifiers/

# --------------------------------------------------------------------------------
# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# --------------------------------------------------------------------------------
# THE TRAINING AND TESTING

# Load iris dataset
iris = datasets.load_iris()

# Select petal length (index 2) and petal width (index 3)
X = iris.data[:, [2, 3]]  
y = iris.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Training the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)  # Predicting test data labels
accuracy = metrics.accuracy_score(y_test, y_pred) * 100  # Calculate accuracy
num_train_samples = len(y_train)
num_test_samples = len(y_test)
num_correct_predictions = (y_pred == y_test).sum()

print(f'Number of training samples: {num_train_samples}')
print(f'Number of test samples: {num_test_samples}')
print(f'Number of correct predictions: {num_correct_predictions}')
print(f'Accuracy: {accuracy:.2f}%')


# -------------------------------------------------------------------------------
# THE PLOTTING
# Create a mesh grid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict classifications for each point in the mesh grid
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Define  colors for each flower type
colors = ['#E69F00', '#56B4E9', '#009E73']  # Tol's bright color palette
cmap_light = ListedColormap([color + '33' for color in colors])  # Lighter color map for decision boundaries
cmap_strong = ListedColormap(colors)  # Strong color map for data points

# Plot 1: Training data, colored by flower type
plt.figure(figsize=(8, 6))
for i, color in zip([0, 1, 2], colors):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=color, edgecolor='k', s=100, label=f'{iris.target_names[i]} (Train)')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Training Data: Colored by Flower Type')
plt.legend(loc='upper left')
plt.show()

# Plot 2: Decision boundaries with training data, with lighter background colors
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)  # Use lighter colormap for decision boundaries with lower opacity
for i, color in zip([0, 1, 2], colors):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=color, edgecolor='k', s=100, label=f'{iris.target_names[i]} (Train)')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Naive Bayes Classifier - Decision Boundaries with Training Data')
plt.legend(loc='upper left')
plt.show()

# Plot 3: Decision boundaries with test data + accuracy, with lighter background colors
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)  # Use lighter colormap for decision boundaries with lower opacity
for i, color in zip([0, 1, 2], colors):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], color=color, edgecolor='k', s=100, marker='x', label=f'{iris.target_names[i]} (Test)')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title(f'Naive Bayes Classifier - Decision Boundaries with Test Data\nAccuracy: {accuracy:.2f}%')
plt.legend(loc='upper left')
plt.show()

