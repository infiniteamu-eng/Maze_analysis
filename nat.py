# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:50:13 2025

@author: BeiqueLab
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Data
wt_averages = [39, 41, 36, 40, 19]
ko_averages = [56, 52, 51, 53, 53]

# Calculate means
wt_mean = np.mean(wt_averages)
ko_mean = np.mean(ko_averages)

# Calculate 95% confidence intervals
def calculate_ci(data):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + 0.95) / 2., n-1)  # 95% CI
    return ci

wt_ci = calculate_ci(wt_averages)
ko_ci = calculate_ci(ko_averages)

# Create figure
fig, ax = plt.subplots(figsize=(2, 4))

# Create bar plot with black borders
bars = ax.bar(['WT', 'KO'], [wt_mean, ko_mean], 
              color=['gray', 'lightgray'], 
              alpha=0.8, 
              width=0.7,
              edgecolor='black',
              linewidth=2)

# Add error bars for confidence intervals
ax.errorbar(['WT', 'KO'], [wt_mean, ko_mean], 
           yerr=[wt_ci, ko_ci], 
           fmt='none', 
           color='black', 
           capsize=5, 
           capthick=1.5,
           elinewidth=1)

# Add individual data points
x_positions = [0, 1]  # Position of bars
for i, (data, x_pos) in enumerate(zip([wt_averages, ko_averages], x_positions)):
    # Add some jitter to x position for visibility
    x_jitter = np.random.normal(x_pos, 0.05, len(data))
    ax.scatter(x_jitter, data, 
              color='black', 
              s=25, 
              alpha=0.8, 
              zorder=3)

# Customize the plot
ax.set_ylabel('Quantity', fontsize=10)
ax.set_xlabel('Genotype', fontsize=10)
ax.set_title('Quantity of Cells: \nWT vs KO', fontsize=14)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Optional: Save the figure
# plt.savefig('wt_ko_comparison.png', dpi=300, bbox_inches='tight')