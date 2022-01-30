'''
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''


from integrator_analysis_helper_functions import *
import sys,os
# load the finles that you need

cwd           = os.getcwd()
directory     = "/SimulationOutput/exercise1/"
filename_100  = "Q1_step_size_100_phase_indexGCO500_numerical_states.dat"
filename_200  = "Q1_step_size_200_phase_indexGCO500_numerical_states.dat"
filename_error_200_q1 = "Q1_step_size_200_phase_indexGCO500_keplerian_difference.dat"

solution_100step,time    = read_solution(cwd+directory+filename_100) #eiliminate time
solution_200step,time    = read_solution(cwd+directory+filename_200) #eliminate time 
solution_200step_q1,time = read_solution(cwd+directory+filename_error_200_q1) #eliminate time 

error_pos_norm_q1  = np.sqrt(np.sum(solution_200step_q1[:,:3]**2))
# so now only take half the points for the 100 step solution and comapere those

solution_100step=solution_100step[::2,:]

error          = solution_200step-solution_100step
print(np.shape(error[:,:3]))
error_pos_norm_extrapolated = np.sqrt(np.sum(error[:,:3]**2,axis=1))

fig, ax       = plt.subplots(2,1)    

print(error_pos_norm_extrapolated)
print(error_pos_norm_q1)
print(error_pos_norm_extrapolated-error_pos_norm_q1)


ax[0].plot((time-time[0])/60/60,abs(error_pos_norm_extrapolated-error_pos_norm_q1),c = 'red')
ax[0].set_xlabel('time [h]')
ax[0].set_ylabel(r'$\epsilon_{q1q6} \quad  [m]$')
ax[0].set_yscale('log')

ax[1].plot((time-time[0])/60/60,error_pos_norm_extrapolated,c = 'red')
ax[1].set_xlabel('time [h]')
ax[1].set_ylabel(r'$\epsilon_{q6} \quad [m]$')
ax[1].set_yscale('log')
plt.tight_layout()
plt.show()
