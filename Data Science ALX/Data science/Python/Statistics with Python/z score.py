from scipy.stats import norm

# Z-score calculated
z_score = 1.897

# Probability of Z-score being greater than calculated value
probability = 1 - norm.cdf(z_score)

print(probability)
