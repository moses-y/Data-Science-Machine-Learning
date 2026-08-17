import numpy as np
from scipy import stats

student_scores = np.array([55, 62, 67, 58, 63, 59, 61, 68, 60, 64])
mu = 60
x_bar = np.mean(student_scores)
s = np.std(student_scores, ddof=1)
n = len(student_scores)
alpha = 0.05
t_stat = (x_bar - mu) / (s / np.sqrt(n))  # Calculate the t-statistic
p_value = stats.t.cdf(t_stat, df=n-1) if t_stat < 0 else (1 - stats.t.cdf(t_stat, df=n-1))  # Calculate the p-value for the two-tailed test
p_value *= 2  # Adjust for two-tailed test

print("T statistic:", t_stat)
print("P-value:", p_value)
