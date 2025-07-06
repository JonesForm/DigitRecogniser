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

print("\n--- Matrix Indexing and Slicing ---")

# Let's create a bigger matrix to play with
# It has 4 rows and 5 columns, so its shape is (4, 5)
data_matrix = np.array([
    [10, 11, 12, 13, 14],
    [20, 21, 22, 23, 24],
    [30, 31, 32, 33, 34],
    [40, 41, 42, 43, 44]
])

print("Our data matrix:")
print(data_matrix)

# 1. Getting a single element
# We use [row, column] notation. Remember, indexing starts at 0!
# Let's get the element in row 2, column 3 (which is 33)
element = data_matrix[2, 3]
print(f"\n1. Single element at (2, 3): {element}") # Using an f-string to format the output

# 2. Getting an entire row
# To get a whole row, specify the row index and use a colon ':' for the column.
# The colon means "get everything in this dimension".
# Let's get the entire first row (index 0)
row_zero = data_matrix[0, :]
print(f"\n2. The entire first row:\n{row_zero}")

# 3. Getting an entire column
# Similarly, use a colon for the row to get a whole column.
# Let's get the entire fourth column (index 3)
column_three = data_matrix[:, 3]
print(f"\n3. The entire fourth column:\n{column_three}")

# 4. Slicing a "sub-matrix"
# We can grab a smaller grid from inside the matrix.
# Syntax is [start_row:end_row, start_col:end_col]
# Note: The 'end' index is NOT included, just like in Python lists.
# Let's get a 2x2 matrix from the top-left corner.
top_left_corner = data_matrix[0:2, 0:2]
print(f"\n4. A 2x2 sub-matrix from the top-left:\n{top_left_corner}")

# 5. Advanced Slicing: Boolean "Masks"
# This is a very powerful technique. We can select elements based on a condition.
# First, create a "mask" of True/False values.
mask = data_matrix > 30
print(f"\n5a. A boolean mask for elements > 30:\n{mask}")

# Now, use this mask to select only the elements that are True.
# This returns a 1D array of all the matching values.
values_over_30 = data_matrix[mask]
print(f"\n5b. All values from the matrix that are > 30:\n{values_over_30}")