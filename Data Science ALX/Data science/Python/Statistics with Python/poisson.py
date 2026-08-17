from scipy.stats import poisson

# Given mean (λ) of the Poisson distribution
lambda_ = 8

# Calculate the probability of serving exactly 6 customers
probability = poisson.pmf(6, lambda_)

print("Probability:", probability)
