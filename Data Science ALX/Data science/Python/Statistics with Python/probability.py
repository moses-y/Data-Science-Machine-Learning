import numpy as np
from scipy.stats import norm

# Given population parameters
mu = 100  # Population mean
sigma = 15  # Population standard deviation
n = 30  # Sample size

# Calculate the standard error of the mean (SEM)
SEM = sigma / np.sqrt(n)

# Calculate the Z-score for a sample mean of 95
z_score = (95 - mu) / SEM

# Calculate the probability that the sample mean will be less than 95
probability = norm.cdf(z_score)

print("Probability:", probability)
