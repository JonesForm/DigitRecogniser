import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load the Data ---
# TensorFlow's Keras library has the MNIST dataset built-in.
# This function returns two tuples: one for training and one for testing.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# --- 2. Explore the Data's Shape ---
# Let's see what we've loaded.
print("Shape of training images:", x_train.shape)
print("Shape of training labels:", y_train.shape)
print("Shape of a single image:", x_train[0].shape)

# The output of x_train.shape is (60000, 28, 28).
# This means we have 60,000 items, and each one is a 28x28 matrix.
# The output of y_train.shape is (60000,).
# This is a vector of 60,000 labels, one for each image.

# --- 3. Visualise a Single Image ---
# Let's look at the very first image in our training set.

image_index = 123 # Try changing this to 5, 100, or any other number!
single_image = x_train[image_index]
actual_label = y_train[image_index]

print(f"\nThis image is labelled as a '{actual_label}'.")

# We use matplotlib to display our 28x28 matrix as an image.
# `imshow` stands for "image show".
# The `cmap='binary'` argument tells matplotlib to use a simple black and white colour map.
plt.imshow(single_image, cmap='binary')

plt.title(f"Label: {actual_label}") # Set the title of the plot
plt.axis('off') # Turn off the x and y axes for a cleaner look
plt.show() # This command opens a window to display the plot.