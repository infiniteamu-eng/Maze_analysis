# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:16:14 2025

@author: BeiqueLab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 22:53:43 2025

@author: amarpreetdheer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Average signal plot across multiple trials
Plot style similar to the reference image with n=3 trials
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
import os

def load_trial_data(data_directory):
    """
    Load data from all trial CSV files in the directory
    """
    data_dir = Path(data_directory)
    trials_data = {}
    
    # Look for plot_xy_data.csv files
    csv_files = list(data_dir.glob("*_plot_xy_data.csv"))
    
    print(f"Found {len(csv_files)} trial data files:")
    
    for csv_file in csv_files:
        # Extract trial info from filename
        filename = csv_file.stem
        trial_name = filename.replace("_plot_xy_data", "")
        
        print(f"  Loading: {trial_name}")
        
        # Load the data
        try:
            df = pd.read_csv(csv_file)
            
            # Load corresponding event data
            event_file = csv_file.parent / f"{trial_name}_event_markers.csv"
            event_data = None
            if event_file.exists():
                event_df = pd.read_csv(event_file)
                if len(event_df) > 0:
                    event_data = event_df.iloc[0]['time_seconds']
            
            trials_data[trial_name] = {
                'time': df['time_seconds'].values,
                'signal': df['zdFF_signal'].values,
                'event_time': event_data
            }
            
        except Exception as e:
            print(f"    Error loading {csv_file}: {e}")
    
    return trials_data

def align_trials_to_event(trials_data, pre_event_window=15, post_event_window=19):
    """
    Align all trials to the reward event (time = 0)
    """
    aligned_data = {}
    
    for trial_name, trial_data in trials_data.items():
        if trial_data['event_time'] is None:
            print(f"Warning: No event data for {trial_name}, skipping")
            continue
            
        # Get original data
        time = trial_data['time']
        signal = trial_data['signal']
        event_time = trial_data['event_time']
        
        # Create aligned time vector (event at t=0)
        aligned_time = time - event_time
        
        # Find indices for the desired window
        pre_idx = np.where(aligned_time >= -pre_event_window)[0]
        post_idx = np.where(aligned_time <= post_event_window)[0]
        
        if len(pre_idx) > 0 and len(post_idx) > 0:
            window_idx = np.intersect1d(pre_idx, post_idx)
            
            aligned_data[trial_name] = {
                'time_aligned': aligned_time[window_idx],
                'signal_aligned': signal[window_idx],
                'original_event_time': event_time
            }
            
            print(f"  {trial_name}: Aligned {len(window_idx)} points around event")
    
    return aligned_data

def interpolate_to_common_timebase(aligned_data, time_resolution=0.02):
    """
    Interpolate all trials to a common time base for averaging
    """
    # Create common time vector
    common_time = np.arange(-15, 19 + time_resolution, time_resolution)
    
    interpolated_data = {}
    
    for trial_name, trial_data in aligned_data.items():
        time_aligned = trial_data['time_aligned']
        signal_aligned = trial_data['signal_aligned']
        
        # Interpolate signal to common time base
        interpolated_signal = np.interp(common_time, time_aligned, signal_aligned)
        
        interpolated_data[trial_name] = {
            'time_common': common_time,
            'signal_common': interpolated_signal
        }
        
        print(f"  {trial_name}: Interpolated to {len(common_time)} points")
    
    return interpolated_data, common_time

def create_average_plot(interpolated_data, common_time, save_path=None):
    """
    Create average signal plot similar to the reference image
    """
    # Collect all signals for averaging
    all_signals = []
    trial_names = list(interpolated_data.keys())
    
    for trial_data in interpolated_data.values():
        all_signals.append(trial_data['signal_common'])
    
    if len(all_signals) == 0:
        print("Error: No data to plot")
        return None
    
    # Convert to numpy array for easier manipulation
    signals_array = np.array(all_signals)
    
    # Calculate statistics
    mean_signal = np.mean(signals_array, axis=0)
    sem_signal = np.std(signals_array, axis=0) / np.sqrt(len(all_signals))
    
    # Create the plot
    plt.style.use('default')  # Clean style
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    # Plot individual trials (thin, light gray lines)
    # for i, signal in enumerate(all_signals):
    #     ax.plot(common_time, signal, color='lightgray', linewidth=0.8, alpha=0.7)
    
    # Plot average with SEM
    ax.plot(common_time, mean_signal, 'black', linewidth=2, label=f'Average (n={len(all_signals)})')
    ax.fill_between(common_time, 
                    mean_signal - sem_signal, 
                    mean_signal + sem_signal, 
                    color='black', alpha=0.2, label='SEM')
    
    # Add event line (reward)
    ax.axvline(x=0, color='black', linestyle=':', linewidth=2, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add "Reward Event" label above the dotted line
    y_max_for_label = ax.get_ylim()[1] * 0.9  # Position at 90% of the top y-limit
    ax.text(0, y_max_for_label, 'Reward Event', ha='center', va='bottom', 
            fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Formatting
    ax.set_xlabel('Time from reward (s)', fontsize=15)
    ax.set_ylabel('ΔF/F', fontsize=15)
    ax.set_title('GRAB 5-HT in PFC', fontsize=16, pad=20, fontweight='bold')
    
    # Set axis limits similar to reference
    ax.set_xlim([-15,5])
    
    # Force x-axis ticks to be integers only
    x_ticks = np.arange(-15, 6, 5)  # Creates ticks at -15, -10, -5, 0, 5
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(tick)) for tick in x_ticks])
    
    # y_max = np.max(np.abs([mean_signal.min(), mean_signal.max()])) * 1.10
    ax.set_ylim([-0.4, 0.6])
   
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add n= text in corner
    ax.text(0.28, 0.98, f'n = {len(all_signals)}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=18, fontweight='bold')
    
    # Add legend
    # ax.legend(frameon=False, loc='upper right')
    
    # Add trial info text
    # trial_info = '\n'.join([f"• {name.split('_')[1]}_{name.split('_')[2]}" for name in trial_names])
    # ax.text(0.02, 0.98, f'Trials:\n{trial_info}', transform=ax.transAxes, 
    #         verticalalignment='top', fontsize=9, 
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        plot_path = save_dir / 'average_photometry_signal.pdf'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Average plot saved: {plot_path}")
        
        # Also save the averaged data
        avg_data = pd.DataFrame({
            'time_from_reward_s': common_time,
            'mean_signal': mean_signal,
            'sem_signal': sem_signal,
            'n_trials': len(all_signals)
        })
        data_path = save_dir / 'average_signal_data.csv'
        avg_data.to_csv(data_path, index=False)
        print(f"Average data saved: {data_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics (n={len(all_signals)} trials):")
    print(f"Trials included: {', '.join(trial_names)}")
    print(f"Time window: {common_time[0]:.1f}s to {common_time[-1]:.1f}s")
    print(f"Baseline period (-15 to -5s): {mean_signal[(common_time >= -15) & (common_time <= -5)].mean():.4f} ± {sem_signal[(common_time >= -15) & (common_time <= -5)].mean():.4f}")
    print(f"Post-reward period (0 to 10s): {mean_signal[(common_time >= 0) & (common_time <= 10)].mean():.4f} ± {sem_signal[(common_time >= 0) & (common_time <= 10)].mean():.4f}")
    
    return {
        'common_time': common_time,
        'mean_signal': mean_signal,
        'sem_signal': sem_signal,
        'individual_signals': signals_array,
        'trial_names': trial_names
    }

# Main execution
if __name__ == "__main__":
    # Set your data directory (where the CSV files are located)
    data_directory = "/Users/BeiqueLab/Desktop/photometry"
    
    print("🔍 MULTI-TRIAL AVERAGE ANALYSIS")
    print("="*50)
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Directory not found at {data_directory}")
        print("Please check the path to your CSV files.")
    else:
        print(f"Loading data from: {data_directory}")
        
        # Step 1: Load trial data
        trials_data = load_trial_data(data_directory)
        
        if len(trials_data) == 0:
            print("Error: No trial data found!")
        else:
            print(f"\n✓ Loaded {len(trials_data)} trials")
            
            # Step 2: Align trials to reward event
            aligned_data = align_trials_to_event(trials_data)
            
            if len(aligned_data) == 0:
                print("Error: No trials could be aligned!")
            else:
                print(f"\n✓ Aligned {len(aligned_data)} trials to reward event")
                
                # Step 3: Interpolate to common timebase
                interpolated_data, common_time = interpolate_to_common_timebase(aligned_data)
                
                print(f"\n✓ Interpolated all trials to common timebase")
                
                # Step 4: Create average plot
                save_directory = os.path.dirname(data_directory)
                results = create_average_plot(interpolated_data, common_time, save_directory)
                
                if results:
                    print(f"\n🎯 Average analysis complete!")
                    print(f"Results saved in: {save_directory}")
                else:
                    print("\n❌ Plot creation failed")