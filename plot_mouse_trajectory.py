# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:29:42 2022

plots mouse trajectory
requires lib_plot_mouse_trajectory

@author: Kelly
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib


'''plot single traj'''
d = plib.TrialData()
d.Load('2025-08-22', '67', 'probe2')
print('Mouse %s Trial %s'%(d.mouse_number, d.trial))

# pltlib.coords_to_target(exp.r_nose, exp.target)
pltlib.plot_single_traj(d, show_target=True, cropcoords = True, savefig=False)

'''plot single traj hole check'''
# pltlib.plot_hole_checks(exp, crop_at_target = True, time_limit = 'all', savefig=False)

''' plots a single mouse trajectory, cut off at REL target '''
# objs = plib.TrialData()
# objs.Load('2021-06-22', '36', 'Probe2')
# print('Mouse %s Trial %s'%(objs.mouse_number, objs.trial))

# pltlib.plot_single_traj(objs, crop_end_custom = objs.target_reverse, continuous = False, savefig = False)

'''plots single trajectory between two points'''
# exp = plib.TrialData()
# exp.Load('2023-02-13', 82, 'Probe4')
# print('Mouse %s Trial %s'%(exp.mouse_number, exp.trial))

# pltlib.plot_single_traj(exp, crop_interval=(exp.target, exp.target_reverse), savefig = True)


'''plots multiple trajectories and REL'''
# objs = [plib.TrialData() for i in range(2)]
# objs[0].Load('2023-07-07', '85', '1')
# objs[1].Load('2023-07-07', '85', '17')


# for i in objs: print('Mouse %s Trial %s'%(i.mouse_number, i.trial))

# pltlib.plot_multi_traj(objs, align_entrance=True, crop_target= True, continuous = False, savefig = True)

# objs = [plib.TrialData() for i in range(3)]
# objs[0].Load('2022-08-12', '69', '1')
# objs[1].Load('2022-08-12', '69', '11')
# objs[2].Load('2022-08-12', '69', '14')

# for i in objs: print('Mouse %s Trial %s'%(i.mouse_number, i.trial))

# pltlib.plot_multi_traj(objs, align_entrance=True, crop_target = True, crop_rev= False, continuous = False, savefig = False)


'''plots multiple trajectories with custom coordinates'''
# objs = [plib.TrialData() for i in range(2)]
# objs[0].Load('2019-10-07', 14, 'R180 6')
# objs[1].Load('2019-10-07', 14, 'R270 1')

# print(objs[0].protocol_name)

# pltlib.plot_multi_traj(objs, crop_end_custom=[objs[0].target, (2.88, 27.45)], savefig = False)


'''plot multiple trajectories on different graphs'''
# objs = [plib.TrialData() for i in range(8)]
# objs[0].Load('2021-06-22', '36', '19')
# objs[1].Load('2021-06-22', '36', '20')
# objs[2].Load('2021-06-22', '36', '21')
# objs[3].Load('2021-06-22', '36', '22')
# objs[4].Load('2021-06-22', '36', '23')
# objs[5].Load('2021-06-22', '36', '24')
# objs[6].Load('2021-06-22', '36', '25')
# objs[7].Load('2021-06-22', '36', '26')


# for i in objs:
#     pltlib.plot_single_traj(i, crop_end_custom = i.target, savefig=True)
    
'''plots heatmap of two experiments'''
# objs = [plib.TrialData() for i in range(8)] #random entrance
# objs[0].Load('2021-07-16', 37, 'Probe')
# objs[1].Load('2021-07-16', 38, 'Probe')
# objs[2].Load('2021-07-16', 39, 'Probe')
# objs[3].Load('2021-07-16', 40, 'Probe')
# objs[4].Load('2021-11-15', 53, 'Probe')
# objs[5].Load('2021-11-15', 54, 'Probe')
# objs[6].Load('2021-11-15', 55, 'Probe')
# objs[7].Load('2021-11-15', 56, 'Probe')



# (objs := plib.TrialData()).Load('2023-12-18', 102, 'Probe')
# pltlib.plot_heatmap(objs, '5min', False)

# objs = [plib.TrialData() for i in range(4)] #3 local cue, not rotating correctly
# objs[0].Load('2019-12-11', 17, 'Probe')
# objs[1].Load('2019-12-11', 18, 'Probe')
# objs[2].Load('2019-12-11', 19, 'Probe')
# objs[3].Load('2019-12-11', 20, 'Probe')
# objs[0].Load('2021-08-11', 45, 'Probe')
# objs[1].Load('2021-08-11', 46, 'Probe')
# objs[2].Load('2021-08-11', 47, 'Probe')
# objs[3].Load('2021-08-11', 48, 'Probe')


# objs = [plib.TrialData() for i in range(14)]
# objs[0].Load('2023-07-07', 85, 'Reverse')
# objs[1].Load('2023-07-07', 86, 'Reverse')
# objs[2].Load('2023-07-07', 87, 'Reverse')
# objs[3].Load('2023-07-07', 88, 'Reverse')
# objs[4].Load('2023-07-07', 89, 'Reverse')
# objs[5].Load('2023-07-07', 90, 'Probe')

# objs[6].Load('2023-08-15', 91, 'Reverse')
# objs[7].Load('2023-08-15', 92, 'Reverse')
# objs[8].Load('2023-08-15', 93, 'Reverse')
# objs[9].Load('2023-08-15', 94, 'Reverse')
# objs[10].Load('2023-08-15', 95, 'Reverse')
# objs[11].Load('2023-08-15', 96, 'Reverse')
# objs[12].Load('2023-08-15', 97, 'Reverse')
# objs[13].Load('2023-08-15', 98, 'Reverse')

# pltlib.plot_heatmap(objs[13], '5min', True)
