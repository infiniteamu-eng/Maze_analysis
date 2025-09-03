#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:43:33 2025

@author: amarpreetdheer
"""

import matplotlib.pyplot as plt
import numpy as np
import modules.calc_latency_distance_speed as calc
# import modules.lib_plot_learning_stats as ls

# Load the ATRX experiment data for all 8 mice (60-67) across all trials
ATRX_trial = calc.iterate_all_trials(['2025-08-22'], training_trials_only=False, continuous=False, show_load=False)

# Define groups based on mice numbers
# KO group: mice 60, 61, 62, 63
# WT group: mice 64, 65, 66, 67
KO_mice = ['60', '61', '62', '63']
WT_mice = ['64', '65', '66', '67']

# Extract data for each measurement type and SLICE TO TRIALS 1-15 ONLY
KO_latency = ATRX_trial['Latency'][KO_mice].iloc[:15]  # Only trials 1-15
WT_latency = ATRX_trial['Latency'][WT_mice].iloc[:15]  # Only trials 1-15

KO_distance = ATRX_trial['Distance'][KO_mice].iloc[:15]  # Only trials 1-15
WT_distance = ATRX_trial['Distance'][WT_mice].iloc[:15]  # Only trials 1-15

KO_speed = ATRX_trial['Speed'][KO_mice].iloc[:15]  # Only trials 1-15
WT_speed = ATRX_trial['Speed'][WT_mice].iloc[:15]  # Only trials 1-15

print("Data loaded successfully!")
print(f"Number of trials: {len(KO_latency)}")
print(f"KO mice: {KO_mice}")
print(f"WT mice: {WT_mice}")

# Calculate means and SEMs for plotting
trials = range(1, len(KO_latency) + 1)  # This will now be 1-15

# Function to calculate mean and SEM
def calc_mean_sem(data):
    means = data.mean(axis=1)
    sems = data.sem(axis=1)
    return means, sems

# PLOT 1: LATENCY
plt.figure(figsize=(6, 4))

ko_lat_mean, ko_lat_sem = calc_mean_sem(KO_latency)
wt_lat_mean, wt_lat_sem = calc_mean_sem(WT_latency)

plt.errorbar(trials, wt_lat_mean, yerr=wt_lat_sem, 
             color='black', marker='o', linewidth=3, markersize=6, 
             capsize=3, label='WT (Wild Type) - n=4')
plt.errorbar(trials, ko_lat_mean, yerr=ko_lat_sem,
             color='#306ed1', marker='o', linewidth=3, markersize=6,
             capsize=3, label='KO (Knockout) - n=4')

plt.title('Latency Learning Curve', fontsize=14, fontweight='bold')
plt.xlabel('Trials', fontsize=12, fontweight='bold')
plt.ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(False)
plt.ylim(bottom=0)
plt.xticks(trials, fontsize=10)
plt.yticks(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# PLOT 2: DISTANCE
plt.figure(figsize=(6, 4))

ko_dist_mean, ko_dist_sem = calc_mean_sem(KO_distance)
wt_dist_mean, wt_dist_sem = calc_mean_sem(WT_distance)

plt.errorbar(trials, wt_dist_mean, yerr=wt_dist_sem,
             color='black', marker='o', linewidth=3, markersize=6,
             capsize=3, label='WT (Wild Type) - n=4')
plt.errorbar(trials, ko_dist_mean, yerr=ko_dist_sem,
             color='#306ed1', marker='o', linewidth=3, markersize=6,
             capsize=3, label='KO (Knockout) - n=4')

plt.title('Distance Learning Curve', fontsize=14, fontweight='bold')
plt.xlabel('Trials', fontsize=12, fontweight='bold')
plt.ylabel('Distance (cm)', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(False)
plt.ylim(bottom=0)
plt.xticks(trials, fontsize=10)
plt.yticks(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# PLOT 3: SPEED
plt.figure(figsize=(6, 4))

ko_speed_mean, ko_speed_sem = calc_mean_sem(KO_speed)
wt_speed_mean, wt_speed_sem = calc_mean_sem(WT_speed)

plt.errorbar(trials, wt_speed_mean, yerr=wt_speed_sem,
             color='black', marker='o', linewidth=3, markersize=6,
             capsize=3, label='WT (Wild Type) - n=4')
plt.errorbar(trials, ko_speed_mean, yerr=ko_speed_sem,
             color='#306ed1', marker='o', linewidth=3, markersize=6,
             capsize=3, label='KO (Knockout) - n=4')

plt.title('Speed Learning Curve', fontsize=14, fontweight='bold')
plt.xlabel('Trials', fontsize=12, fontweight='bold')
plt.ylabel('Speed (cm/s)', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(False)
plt.ylim(bottom=0)
plt.xticks(trials, fontsize=10)
plt.yticks(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Print summary statistics (for trials 1-15 only)
print("\nSummary Statistics (Trials 1-15):")
print("=" * 60)
print(f"Latency - WT mean: {WT_latency.mean().mean():.2f}s, KO mean: {KO_latency.mean().mean():.2f}s")
print(f"Distance - WT mean: {WT_distance.mean().mean():.2f}cm, KO mean: {KO_distance.mean().mean():.2f}cm")  
print(f"Speed - WT mean: {WT_speed.mean().mean():.2f}cm/s, KO mean: {KO_speed.mean().mean():.2f}cm/s")