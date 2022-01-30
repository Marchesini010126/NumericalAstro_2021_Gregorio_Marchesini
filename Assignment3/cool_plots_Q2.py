import matplotlib.pyplot as plt
import os
import numpy as np
from integrator_analysis_helper_functions import read_solution

#obtain list of interesting files
dir_name   = 'SimulationOutput/exercise2'
files_list = os.listdir(dir_name)
# only take useful files and elimnate directories
files_list = [file for file in files_list if 'dat' in file.split('.')]
# define names of the tolerances
integration_tolerances_names = ['10n12','10n10','10n8','10n6']
phase_names                  = ["FLYBY","GCO500"] 


fig1,ax1 = plt.subplots(2,1)
fig2,ax2 = plt.subplots(2,1)


for file in files_list :
    
    file_markers = file.split('_')
    if "FLYBY" in file_markers  and "benchmark" in file_markers :
        
        for tol in integration_tolerances_names :
            if tol in file_markers:
                power_tol = tol.split('n')[1]
        
        error_from_banchmark,time_stamps = read_solution(os.path.join(dir_name,file))
        time_step = np.diff(time_stamps)
        
        position_error = np.sqrt(np.sum(error_from_banchmark[:,:3]**2,axis=1))
        
        ax1[0].plot((time_stamps-time_stamps[0])/60/60,position_error,label='10E-'+power_tol)
        ax1[0].set_yscale('log')
        ax1[0].set_xlabel('time [h]')
        ax1[0].set_ylabel(r'$\epsilon (t) [m]$')
    
        ax1[1].plot((time_stamps[:-1]-time_stamps[0])/60/60,time_step,label='10E-'+power_tol)
        ax1[1].set_yscale('log')
        ax1[1].set_xlabel('time [h]')
        ax1[1].set_ylabel(r'$\Delta t [s]$')
    
        
        
    elif "GCO500" in file_markers  and "benchmark" in file_markers :
        for tol in integration_tolerances_names :
            if tol in file_markers:
                power_tol = tol.split('n')[1]
        
        error_from_banchmark,time_stamps = read_solution(os.path.join(dir_name,file))
        
        time_step = np.diff(time_stamps)
        position_error = np.sqrt(np.sum(error_from_banchmark[:,:3]**2,axis=1))
        
        ax2[0].plot((time_stamps-time_stamps[0])/60/60,position_error,label='10E-'+power_tol)
        ax2[0].set_yscale('log')
        ax2[0].set_xlabel('time [h]')
        ax2[0].set_ylabel(r'$\epsilon (t) [m]$')
        
        ax2[1].plot((time_stamps[:-1]-time_stamps[0])/60/60,time_step,label='10E-'+power_tol)
        #ax2[1].plot((time_stamps[:40]-time_stamps[0])/60/60,time_step[:40],label='10E-'+power_tol)
        
        ax2[1].set_yscale('log')
        ax2[1].set_xlabel('time [h]')
        ax2[1].set_ylabel(r'$\Delta t [s]$')

fig1.tight_layout()
fig1.suptitle('FLYBY')
fig1.subplots_adjust(top=0.88)

fig2.tight_layout()
fig2.suptitle('GCO500')
fig2.subplots_adjust(top=0.88)

for axes1,axes2 in zip(ax1,ax2) :
    axes1.legend()
    axes2.legend()
    
plt.show()





