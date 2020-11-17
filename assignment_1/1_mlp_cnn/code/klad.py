import numpy as np

# A numpy array with values between 0 and 5, with 1000 rows and 100 columns
arr = np.random.randint(0, 5, (1000,100))

# Column means. THIS CALCULATES ALONG THE ROWS, HENCE AXIS=0
col_means = np.mean(arr, axis=0)

# Check that is has one value for each column: 100
assert col_means.shape == (100,)

# Gives two arrays: the rows, and the columsn that match the condition.
# Take the second, find the corresponding means, and set the 0 values to this
arr[arr == 0] = col_means[np.where(arr == 0)[1]]

# Column means. THIS CALCULATES ALONG THE ROWS, HENCE AXIS=0
col_means = np.mean(arr, axis=0)
col_stds = np.std(arr, axis=0)

arr = (arr - col_means[None,:]) / col_stds[None,:]

# Check that everything is close to 0 mean and 1 std
assert np.allclose(np.mean(arr, axis=0), 0)
assert np.allclose(np.std(arr, axis=0),  1)
