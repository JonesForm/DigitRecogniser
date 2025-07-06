import tensorflow as tf
import numpy as np

# --- 1. Load the Data ---
# We still use tensorflow to conveniently load the MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# --- 2. Data Preprocessing ---

# -- FLATTENING --
# We reshape the 60,000 x 28 x 28 matrices into 60,000 x 784 vectors.
print("Shape of x_train before flattening:", x_train.shape)

num_train_samples = x_train.shape[0]
num_test_samples = x_test.shape[0]

x_train = x_train.reshape((num_train_samples, 28 * 28))
x_test = x_test.reshape((num_test_samples, 28 * 28))

print("Shape of x_train after flattening:", x_train.shape)

# -- NORMALISATION --
# We convert the pixel values from integers (0-255) to floats (0.0-1.0).
# This is a crucial step for most machine learning models.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("\nData types after normalisation:", x_train.dtype)
print("Example pixel values after normalisation:", x_train[0, 350:360]) # Print a small slice

# Our data is now prepped and ready for a machine learning model!
print("\nData preprocessing complete.")

# --- 3. Training the Model ---
# We need to import the model from scikit-learn.
# We'll use SGDClassifier, which stands for Stochastic Gradient Descent Classifier.
# It's a simple and efficient model for classification tasks.
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt # We'll use this to view our test image

print("\n--- Model Training ---")

# Create an instance of the model. random_state makes the results reproducible.
model = SGDClassifier(random_state=42)

# This is the LEARNING step.
# We "fit" the model using our training images (x_train) and their correct labels (y_train).
# The model will analyse this data to learn the patterns that define each digit.
print("Training the model... (This may take a few seconds)")
model.fit(x_train, y_train)
print("Training complete!")


# --- 4. Making a Prediction ---
print("\n--- Model Prediction ---")

# Let's grab a single image from the TEST set, which the model has NEVER seen before.
test_image_index = 123
some_digit = x_test[test_image_index]
actual_label = y_test[test_image_index]

# Use our trained model to predict what this digit is.
# The model expects a 2D array, so we wrap `some_digit` in square brackets.
prediction = model.predict([some_digit])

print(f"The model predicts this digit is a: {prediction[0]}")
print(f"The actual label is: {actual_label}")

# Let's visualise the digit we tested
# We have to reshape it back from a 784-element vector to a 28x28 matrix to view it.
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap='binary')
plt.title(f"Prediction: {prediction[0]} | Actual: {actual_label}")
plt.axis('off')
plt.show()

# --- 5. Evaluating the Model ---
# We will use cross-validation to get a robust measure of our model's accuracy.
from sklearn.model_selection import cross_val_score

print("\n--- Model Evaluation ---")
print("Performing 3-fold cross-validation... (This will take a bit longer)")

# cross_val_score automates the process described above.
# We give it:
# - our model object
# - the training data (it uses the training data for this, not the test data)
# - the training labels
# - cv=3: The number of folds to use.
# - scoring="accuracy": The metric we want to measure.
scores = cross_val_score(model, x_train, y_train, cv=3, scoring="accuracy")

print("Cross-validation complete.")
print("Accuracy scores for each of the 3 folds:", scores)
print(f"Average accuracy: {scores.mean():.4f}") # .mean() calculates the average