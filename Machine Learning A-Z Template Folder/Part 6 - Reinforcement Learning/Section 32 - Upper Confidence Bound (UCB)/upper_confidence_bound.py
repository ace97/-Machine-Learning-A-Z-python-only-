# Upper Confidence Bound
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import math
# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N = 10000
d = 10
ads_select = []
numbers_selections = [0] * d
sums_of_reward = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    reward = 0
    for i in range(0, d):
        if numbers_selections[i] > 0:
            average_reward = sums_of_reward[i] / numbers_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_select.append(ad)
    numbers_selections[ad] = numbers_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_reward[ad] = sums_of_reward[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_select)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
