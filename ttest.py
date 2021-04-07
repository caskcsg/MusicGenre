

## Import the packages
import numpy as np


# ## Define 2 random distributions
# #Sample Size
# N = 10
# #Gaussian distributed data with mean = 2 and var = 1
# a = np.random.randn(N) + 2
# #Gaussian distributed data with with mean = 0 and var = 1
# b = np.random.randn(N)
#
#
# ## Calculate the Standard Deviation
# #Calculate the variance to get the standard deviation
#
# #For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
# var_a = a.var(ddof=1)
# var_b = b.var(ddof=1)
#
# #std deviation
# s = np.sqrt((var_a + var_b)/2)
#
# ## Calculate the t-statistics
# t = (a.mean() - b.mean())/(s*np.sqrt(2/N))
#
# ## Compare with the critical t-value
# #Degrees of freedom
# df = 2*N - 2
#
#
#
#
# #p-value after comparison with the t
# p = 1 - stats.t.cdf(t,df=df)
#
#
# print("t = " + str(t))
# print("p = " + str(2*p))
### You can see that after comparing the t statistic with the critical t value (computed internally)
# we get a good p value of 0.0005 and thus we reject the null hypothesis and
# thus it proves that the mean of the two distributions are different and statistically significant.




# b = [40.3,  54.7,  81.9,  39.0,  42.2,  20.8,  11.8,  22.3,  58.9,  32.5,  46.3, 77.5]
# b = [80.6, 89.6, 98.6, 58.1, 62.7, 47.8, 29.6, 48.5, 79.3, 59.8, 79.1,  95.3]

## Cross Checking with the internal scipy function
# t2, p2 = stats.ttest_rel(a, b)
# print("t = " + str(t2))
# print("p = " + str(p2))
from scipy import stats
SOTA = [78.6, 88.6, 97.6,  56.8, 74.1,  92.3]  # 57.1, 61.7, 44.8, 27.6, 43.5, 78.3,
Our = [80.1, 89.8, 97.6,  60.6, 78.6,  94.7]  # 58.9, 64.3, 47.8, 29.3, 46.5, 80.8,

print(stats.ttest_rel(SOTA, Our))

# Ttest_relResult(statistic=-6.71318773997175, pvalue=3.316902032898069e-05)

# Ttest_relResult(statistic=-5.777456944569636, pvalue=0.028676519553306034)

# 0.00003316



