import numpy as np

# Set standard deviation
sigma = 1  # You can change this to any other value

# Define the covariance matrix
cov_matrix = np.array([[2 * sigma**2, -sigma**2],
                       [-sigma**2, 2 * sigma**2]])

# Define the mean vector (zero mean)
mean = np.array([0, 0])

# Generate samples from the multivariate normal distribution
# Sample size: Let's say we want 1000 samples
n_samples = 1000
samples = np.random.multivariate_normal(mean, cov_matrix, n_samples)

# 'samples' is an array of shape (n_samples, 2) where each row is a (Y, Z) pair
print(samples[:5])  # Print the first 5 samples for verification