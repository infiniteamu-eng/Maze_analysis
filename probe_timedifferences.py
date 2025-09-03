import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt

# Turn off interactive mode to prevent automatic showing
plt.ioff()

# First, get target A coordinates from probe1 trial
d_probe1 = plib.TrialData()
d_probe1.Load('2025-08-22', '60', 'probe1')
target_A_coords = d_probe1.target
print(f'Target A coordinates: {target_A_coords}')

# Load probe2 data
d_probe2 = plib.TrialData()
d_probe2.Load('2025-08-22', '60', 'probe2')
print(f'Mouse {d_probe2.mouse_number} Trial {d_probe2.trial}')
print(f'Target B coordinates: {d_probe2.target}')

# Create the plot
fig, ax = plt.subplots()

# Manually recreate the essential parts of plot_single_traj to avoid plt.show()
# Draw arena if available
if hasattr(d_probe2, 'arena_circle'):
    pltlib.draw_arena(d_probe2, ax)
else:
    print('No arena circle found')

# Get trajectory endpoints based on which target is visited first
target_A_index = None
target_B_index = None

try:
    target_A_index = pltlib.coords_to_target(d_probe2.r_nose, target_A_coords)
except:
    pass

try:
    target_B_index = pltlib.coords_to_target(d_probe2.r_nose, d_probe2.target)
except:
    pass

# Determine plotting endpoint
if target_A_index is not None and target_B_index is not None:
    if target_A_index < target_B_index:
        # Mouse goes to A first, plot until B
        plot_end_index = target_B_index
        plot_strategy = "A first, then B"
    else:
        # Mouse goes to B first, plot until A  
        plot_end_index = target_A_index
        plot_strategy = "B first, then A"
elif target_A_index is not None:
    plot_end_index = target_A_index
    plot_strategy = "A only"
elif target_B_index is not None:
    plot_end_index = target_B_index  
    plot_strategy = "B only"
else:
    plot_end_index = len(d_probe2.r_nose) - 1
    plot_strategy = "neither target reached"

print(f'Plotting strategy: {plot_strategy}')

# Plot trajectory up to the final target visit
ax.plot(d_probe2.r_nose[:plot_end_index+1,0], d_probe2.r_nose[:plot_end_index+1,1], ls='-', color='#004E89')

# Add target B (blue circle - current target)
target_B_circle = plt.Circle(d_probe2.target, 2.5, color='b')
ax.add_artist(target_B_circle)

# Add orange circle for target A (previous reward location)
target_A_circle = plt.Circle(target_A_coords, 2.5, color='orange', fill=True, linewidth=2)
ax.add_artist(target_A_circle)

# Add text label
ax.text(target_A_coords[0], target_A_coords[1]-5, 'A', 
        ha='center', va='top', color='orange', fontweight='bold')

# Draw entrance
pltlib.draw_entrance(d_probe2, ax)

ax.axis('off')
ax.set_aspect('equal')

# Now show the complete plot
plt.show()

# Turn interactive mode back on
plt.ion()

# Check which target the mouse reaches first and analyze search strategy
try:
    target_A_index = pltlib.coords_to_target(d_probe2.r_nose, target_A_coords)
    target_A_time = d_probe2.time[target_A_index]
    print(f'Mouse reached target A at index: {target_A_index}, time: {target_A_time:.2f} seconds')
except:
    target_A_index = None
    target_A_time = None
    print('Mouse did not reach target A location')

try:
    target_B_index = pltlib.coords_to_target(d_probe2.r_nose, d_probe2.target)
    target_B_time = d_probe2.time[target_B_index]
    print(f'Mouse reached target B at index: {target_B_index}, time: {target_B_time:.2f} seconds')
except:
    target_B_index = None
    target_B_time = None
    print('Mouse did not reach target B location')

# Determine search strategy and calculate time differences
if target_A_index is not None and target_B_index is not None:
    time_between_targets = abs(target_B_time - target_A_time)
    if target_A_index < target_B_index:
        print(f'Search strategy: A first (t={target_A_time:.2f}s), then B (t={target_B_time:.2f}s)')
        print(f'Time between A and B: {time_between_targets:.2f} seconds')
        print('Mouse checked previous reward location first, then current target!')
    else:
        print(f'Search strategy: B first (t={target_B_time:.2f}s), then A (t={target_A_time:.2f}s)')
        print(f'Time between B and A: {time_between_targets:.2f} seconds')
        print('Mouse went to current target first, then checked previous reward location!')
elif target_A_index is not None:
    print(f'Mouse only visited target A at {target_A_time:.2f} seconds (previous reward location)')
elif target_B_index is not None:
    print(f'Mouse only visited target B at {target_B_time:.2f} seconds (current target location)')
else:
    print('Mouse did not reach either target location')