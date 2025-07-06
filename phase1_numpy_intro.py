# JonesForm
import numpy as np

# The 'as np' is a standard convention that lets us type 'np' instead of 'numpy'.

# Create our first 2D array (a 3x4 matrix)
my_matrix = np.array([
    [5, 8, 10, 12],
    [4, 3, 2, 1],
    [0, 6, 9, 11]
])

# Print the matrix to the console
print("Our first matrix:")
print(my_matrix)

# Print the 'shape' of the matrix (rows, columns)
print("\nThe shape of our matrix is:")
print(my_matrix.shape)

# Print the data type of the elements in the matrix
print("\nThe data type of the elements is:")
print(my_matrix.dtype)

# Continuing in phase1_numpy_intro.py

print("\n--- Scalar Operations ---")
my_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Add 100 to every element
result_add = my_matrix + 100
print("Matrix after adding 100:")
print(result_add)

# Multiply every element by 2
result_mul = my_matrix * 2
print("\nMatrix after multiplying by 2:")
print(result_mul)

print("\n--- Element-wise Operations ---")
matrix_a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

matrix_b = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

# Add the two matrices
result_add_matrices = matrix_a + matrix_b
print("Result of adding matrix_a and matrix_b:")
print(result_add_matrices)

print("\n--- The Dot Product ---")
# matrix_a has shape (2, 3)
matrix_a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# matrix_c has shape (3, 2)
matrix_c = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

# The inner dimensions match (3 and 3), so we can perform the dot product.
# The result will have shape (2, 2).
dot_product_result = matrix_a @ matrix_c
print("Dot product of matrix_a (2x3) and matrix_c (3x2):")
print(dot_product_result)
print("\nShape of the result:", dot_product_result.shape)