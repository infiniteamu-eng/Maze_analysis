#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:43:03 2025

@author: amarpreetdheer
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import pandas as pd

# Define mouse groups
KO_mice = ['60', '61', '62', '63']
WT_mice = ['64', '65', '66', '67']

def calculate_spatial_entropy(coords, n_bins=20):
    """
    Calculate spatial entropy based on how uniformly space is explored
    Higher entropy = more uniform exploration
    Lower entropy = more focused/clustered exploration
    """
    # Remove NaN values
    valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
    coords_clean = coords[valid_idx]
    
    if len(coords_clean) < 10:
        return np.nan
    
    # Create 2D histogram of spatial occupancy
    x_range = [coords_clean[:, 0].min(), coords_clean[:, 0].max()]
    y_range = [coords_clean[:, 1].min(), coords_clean[:, 1].max()]
    
    # Ensure reasonable range
    if x_range[1] - x_range[0] < 1:
        x_range = [x_range[0] - 5, x_range[1] + 5]
    if y_range[1] - y_range[0] < 1:
        y_range = [y_range[0] - 5, y_range[1] + 5]
    
    H, x_edges, y_edges = np.histogram2d(coords_clean[:, 0], coords_clean[:, 1], 
                                        bins=n_bins, 
                                        range=[x_range, y_range])
    
    # Convert to probability distribution
    H_flat = H.flatten()
    H_flat = H_flat[H_flat > 0]  # Remove empty bins
    prob_dist = H_flat / np.sum(H_flat)
    
    # Calculate Shannon entropy
    spatial_entropy = entropy(prob_dist, base=2)
    
    return spatial_entropy

def calculate_movement_entropy(coords, n_bins=16):
    """
    Calculate movement direction entropy
    Higher entropy = more varied movement directions
    Lower entropy = more consistent movement direction
    """
    valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
    coords_clean = coords[valid_idx]
    
    if len(coords_clean) < 3:
        return np.nan
    
    # Calculate movement vectors
    movement_vectors = np.diff(coords_clean, axis=0)
    
    # Calculate angles
    angles = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0])
    
    # Convert to degrees and normalize to [0, 360)
    angles_deg = np.degrees(angles) % 360
    
    # Create histogram of movement directions
    hist, _ = np.histogram(angles_deg, bins=n_bins, range=(0, 360))
    
    # Convert to probability distribution
    hist = hist[hist > 0]  # Remove empty bins
    prob_dist = hist / np.sum(hist)
    
    # Calculate entropy
    movement_entropy = entropy(prob_dist, base=2)
    
    return movement_entropy

def calculate_sequence_entropy(coords, step_size=5):
    """
    Calculate behavioral sequence entropy based on spatial transitions
    Divides arena into regions and analyzes transition patterns
    """
    valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
    coords_clean = coords[valid_idx]
    
    if len(coords_clean) < 20:
        return np.nan
    
    # Divide arena into spatial regions (quadrants or grid)
    x_center = np.median(coords_clean[:, 0])
    y_center = np.median(coords_clean[:, 1])
    
    # Assign each coordinate to a region (0=SW, 1=SE, 2=NW, 3=NE)
    regions = []
    for coord in coords_clean:
        if coord[0] < x_center and coord[1] < y_center:
            regions.append(0)  # SW
        elif coord[0] >= x_center and coord[1] < y_center:
            regions.append(1)  # SE
        elif coord[0] < x_center and coord[1] >= y_center:
            regions.append(2)  # NW
        else:
            regions.append(3)  # NE
    
    regions = np.array(regions)
    
    # Sample regions at regular intervals
    sampled_regions = regions[::step_size]
    
    if len(sampled_regions) < 3:
        return np.nan
    
    # Create transition sequences (bigrams)
    transitions = []
    for i in range(len(sampled_regions) - 1):
        transitions.append((sampled_regions[i], sampled_regions[i + 1]))
    
    # Count transition frequencies
    unique_transitions, counts = np.unique(transitions, axis=0, return_counts=True)
    
    # Convert to probability distribution
    prob_dist = counts / np.sum(counts)
    
    # Calculate entropy
    sequence_entropy = entropy(prob_dist, base=2)
    
    return sequence_entropy

def analyze_mouse_entropy(mouse_id, trials=range(11, 16)):
    """Analyze entropy measures for a single mouse across specified trials"""
    
    results = []
    
    for trial in trials:
        try:
            # Load trial data
            d = plib.TrialData()
            d.Load('2025-08-22', mouse_id, str(trial))
            
            # Get coordinates up to target
            coords = d.r_nose
            try:
                target_idx = pltlib.coords_to_target(coords, d.target)
                coords_to_analyze = coords[:target_idx+1]
            except:
                coords_to_analyze = coords
            
            # Calculate different entropy measures
            spatial_ent = calculate_spatial_entropy(coords_to_analyze)
            movement_ent = calculate_movement_entropy(coords_to_analyze)
            sequence_ent = calculate_sequence_entropy(coords_to_analyze)
            
            results.append({
                'mouse_id': mouse_id,
                'trial': trial,
                'spatial_entropy': spatial_ent,
                'movement_entropy': movement_ent,
                'sequence_entropy': sequence_ent
            })
            
        except Exception as e:
            print(f"Error analyzing mouse {mouse_id}, trial {trial}: {e}")
            continue
    
    return results

def plot_entropy_comparison(trials=range(11, 16)):
    """Compare entropy measures between KO and WT groups"""
    
    # Collect data for all mice
    all_results = []
    
    for mouse_id in KO_mice + WT_mice:
        mouse_results = analyze_mouse_entropy(mouse_id, trials)
        all_results.extend(mouse_results)
    
    if not all_results:
        print("No data available for entropy analysis")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Add group information
    df['group'] = df['mouse_id'].apply(lambda x: 'KO' if x in KO_mice else 'WT')
    
    # Calculate mean entropy per mouse
    mouse_summary = df.groupby(['mouse_id', 'group']).agg({
        'spatial_entropy': 'mean',
        'movement_entropy': 'mean',
        'sequence_entropy': 'mean'
    }).reset_index()
    
    print("Entropy Analysis Results:")
    print(mouse_summary)
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    entropy_measures = [
        ('spatial_entropy', 'Spatial Entropy\n(Exploration Uniformity)', ax1),
        ('movement_entropy', 'Movement Entropy\n(Directional Variability)', ax2),
        ('sequence_entropy', 'Sequence Entropy\n(Behavioral Transitions)', ax3)
    ]
    
    for measure, title, ax in entropy_measures:
        ko_data = mouse_summary[mouse_summary['group'] == 'KO'][measure].dropna().values
        wt_data = mouse_summary[mouse_summary['group'] == 'WT'][measure].dropna().values
        
        # Individual points
        ko_x = np.random.normal(1, 0.05, len(ko_data))
        wt_x = np.random.normal(2, 0.05, len(wt_data))
        
        ax.scatter(ko_x, ko_data, alpha=0.7, s=60, color='#306ed1', label='KO')
        ax.scatter(wt_x, wt_data, alpha=0.7, s=60, color='black', label='WT')
        
        # Group means
        if len(ko_data) > 0 and len(wt_data) > 0:
            ko_mean, wt_mean = np.mean(ko_data), np.mean(wt_data)
            ko_sem, wt_sem = stats.sem(ko_data), stats.sem(wt_data)
            
            ax.errorbar(1, ko_mean, yerr=ko_sem, fmt='o', markersize=8, color='#306ed1',
                       capsize=5, capthick=2, elinewidth=2, markeredgecolor='white')
            ax.errorbar(2, wt_mean, yerr=wt_sem, fmt='o', markersize=8, color='black',
                       capsize=5, capthick=2, elinewidth=2, markeredgecolor='white')
            
            # Statistics
            t_stat, p_value = stats.ttest_ind(ko_data, wt_data)
            sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else f"p={p_value:.3f}"
            
            y_max = max(max(ko_data), max(wt_data))
            ax.plot([1, 2], [y_max*1.1, y_max*1.1], 'k-', linewidth=1)
            ax.text(1.5, y_max*1.12, sig_text, ha='center', va='bottom')
        
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['KO', 'WT'])
        ax.set_ylabel('Entropy (bits)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Fourth subplot: entropy correlation analysis
    spatial_data = mouse_summary['spatial_entropy'].dropna().values
    movement_data = mouse_summary['movement_entropy'].dropna().values
    
    if len(spatial_data) > 0 and len(movement_data) > 0:
        colors = ['#306ed1' if g == 'KO' else 'black' for g in mouse_summary['group'] 
                 if not np.isnan(mouse_summary[mouse_summary['group']==g]['spatial_entropy']).any()]
        
        ax4.scatter(spatial_data, movement_data, c=colors, alpha=0.7, s=60)
        ax4.set_xlabel('Spatial Entropy')
        ax4.set_ylabel('Movement Entropy') 
        ax4.set_title('Entropy Relationship')
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # Add correlation
        if len(spatial_data) == len(movement_data):
            r, p = stats.pearsonr(spatial_data, movement_data)
            ax4.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle(f'Entropy Analysis - Navigation Patterns (Trials {trials[0]}-{trials[-1]})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nEntropy Summary (Trials {trials[0]}-{trials[-1]}):")
    
    for measure in ['spatial_entropy', 'movement_entropy', 'sequence_entropy']:
        ko_data = mouse_summary[mouse_summary['group'] == 'KO'][measure].dropna().values
        wt_data = mouse_summary[mouse_summary['group'] == 'WT'][measure].dropna().values
        
        if len(ko_data) > 0 and len(wt_data) > 0:
            print(f"\n{measure.replace('_', ' ').title()}:")
            print(f"  KO: {np.mean(ko_data):.3f} ± {stats.sem(ko_data):.3f} bits")
            print(f"  WT: {np.mean(wt_data):.3f} ± {stats.sem(wt_data):.3f} bits")
            
            t_stat, p_value = stats.ttest_ind(ko_data, wt_data)
            print(f"  t-test: t={t_stat:.3f}, p={p_value:.3f}")

def plot_entropy_heatmaps(mouse_id, trial=15):
    """Visualize spatial entropy for a single mouse/trial"""
    
    try:
        d = plib.TrialData()
        d.Load('2025-08-22', mouse_id, str(trial))
        
        coords = d.r_nose
        valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
        coords_clean = coords[valid_idx]
        
        if len(coords_clean) < 10:
            print(f"Insufficient data for mouse {mouse_id}, trial {trial}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        group = 'KO' if mouse_id in KO_mice else 'WT'
        
        # Plot 1: Trajectory
        ax1.plot(coords_clean[:, 0], coords_clean[:, 1], 'b-', alpha=0.6)
        ax1.scatter(coords_clean[0, 0], coords_clean[0, 1], color='green', s=100, marker='o')
        ax1.scatter(d.target[0], d.target[1], color='red', s=100, marker='s')
        ax1.set_aspect('equal')
        ax1.set_title(f'Mouse {mouse_id} ({group}) - Trial {trial}')
        
        # Plot 2: Spatial occupancy heatmap
        H, x_edges, y_edges = np.histogram2d(coords_clean[:, 0], coords_clean[:, 1], bins=20)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        
        im = ax2.imshow(H.T, extent=extent, origin='lower', cmap='viridis')
        ax2.set_aspect('equal')
        ax2.set_title('Spatial Occupancy Density')
        plt.colorbar(im, ax=ax2)
        
        # Calculate and display entropy
        spatial_ent = calculate_spatial_entropy(coords_clean)
        movement_ent = calculate_movement_entropy(coords_clean)
        
        fig.suptitle(f'Spatial Entropy: {spatial_ent:.3f} bits | Movement Entropy: {movement_ent:.3f} bits')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating heatmap for mouse {mouse_id}: {e}")

# Run entropy analysis
if __name__ == "__main__":
    print("Analyzing spatial navigation entropy...")
    
    # Compare entropy between groups for trials 11-15
    plot_entropy_comparison(trials=range(11, 16))
    
    # Example entropy heatmaps
    print("\nGenerating example entropy visualizations...")
    plot_entropy_heatmaps('60', 15)  # KO example
    plot_entropy_heatmaps('67', 15)  # WT example