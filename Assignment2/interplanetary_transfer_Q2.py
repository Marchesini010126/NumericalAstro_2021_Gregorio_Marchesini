''' 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

from interplanetary_transfer_helper_functions import *
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 2 #################################################
###########################################################################

# initial settings
frame_origin         = 'Sun'
bodies_to_create     = ['Sun','Mars','Earth','Venus','Moon','Jupiter']
vehicle_specific     = {'name':'Spacecraft',   #name
                        'mass':1000,           #kg
                        'radiation_ref_A':20,  #m^2
                        'radiation_coefficient':1.2}   

sun_gravitational_parameter = 1.32712440042E20 #m^3/s^2


# Create body objects
bodies = create_simulation_bodies( bodies_to_create,frame_origin,vehicle_spec=vehicle_specific)

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

"""
case_i  : The initial and final propagation time equal to the initial and final times of the Lambert arc.
case_ii : The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour.
case_iii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=2 days.

"""

def history2array(state_history):
    state = np.array(list(state_history.values()))
    time  = np.array(list(state_history.keys())).reshape((-1,1))
    res = np.hstack((time,state))
    return res

def history2DataFrame(state_history,classes):
    sol= history2array(state_history)
    frame = pd.DataFrame(sol,columns=classes)
    return frame

def array2DataFrame(array,columns) :
    frame = pd.DataFrame(array,columns=columns)
    return frame
    

# List cases to iterate over. STUDENT NOTE: feel free to modify if you see fit
cases       = ['case_i', 'case_ii', 'case_iii']
buffer_list = [r'$\Delta t = 0$',r'$\Delta t = 1 hr$',r'$\Delta t = 2 days$']
buffer_time = [0       ,60*60     ,60*60*24*2]   #s
an_hour_in_seconds = 60*60 #s
# preallocation

simulations_results     = dict()
multi_body_acceleration = dict()
distances               = dict()
time_stumps             = dict()



# Run propagation for each of cases i-iii
for case,delta in zip(cases,buffer_time):

    # Define the initial and final propagation time for the current case
    time_of_flight_with_buffer  = time_of_flight  - 2*delta
    departure_epoch_with_buffer = departure_epoch + delta
    arrival_epoch_with_buffer   = departure_epoch_with_buffer + time_of_flight_with_buffer

    # Perform propagation
    dynamics_simulator = propagate_trajectory( departure_epoch_with_buffer, arrival_epoch_with_buffer, bodies, lambert_arc_ephemeris,
                     use_perturbations = True)
    write_propagation_results_to_file(
        dynamics_simulator, lambert_arc_ephemeris, "Q2a_" + str(cases.index(case)), output_directory)

    state_history           = dynamics_simulator.state_history
    dependant_acc           = dynamics_simulator.dependent_variable_history 
    
    lambert_history               = get_lambert_arc_history(lambert_arc_ephemeris, state_history)
    simulations_results[case]     = {'numerical' :history2array(state_history),'lambert':history2array(lambert_history) } 
    
    multi_body_acceleration[case] =  history2array(dependant_acc)[:,:4]   
    distances[case]               = {'Earth':history2array(dependant_acc)[:,4], 'Venus':history2array(dependant_acc)[:,5]}
    
    time                          = simulations_results['case_i']['numerical'][:,0]
    time                          = (time-time[0])/60/60/24
    time_stumps[case]             = time
    
    if case == 'case_ii' : 
        middle_time = departure_epoch_with_buffer+(time_of_flight-2*delta)/2
        
        # right side
        dynamics_simulator_right = propagate_trajectory(middle_time, middle_time + time_of_flight_with_buffer/2, bodies, lambert_arc_ephemeris,
                     use_perturbations = True)
        write_propagation_results_to_file(
           dynamics_simulator_right, lambert_arc_ephemeris, "Q2a_" + str(cases.index(case)) + '_right_side', output_directory)
        
        state_middle_right   = history2array(dynamics_simulator_right.state_history)
        dependant_acc_right  = history2array(dynamics_simulator_right.dependent_variable_history)[:,:4]
        lambert_state_right  = history2array(get_lambert_arc_history(lambert_arc_ephemeris, dynamics_simulator_right.state_history))
        rel_dist_right       = {'Earth':history2array(dynamics_simulator_right.dependent_variable_history)[:,4], 'Venus':history2array(dynamics_simulator_right.dependent_variable_history)[:,5]} 
        # left side
        dynamics_simulator_left = propagate_trajectory(middle_time,middle_time - time_of_flight_with_buffer/2, bodies, lambert_arc_ephemeris,
                     use_perturbations = True)
        write_propagation_results_to_file(
           dynamics_simulator_left, lambert_arc_ephemeris, "Q2a_" + str(cases.index(case)) + '_left_side', output_directory)
        
        state_middle_left   = history2array(dynamics_simulator_left.state_history)
        dependant_acc_left  = history2array(dynamics_simulator_left.dependent_variable_history)[:,:4]
        lambert_state_left  = history2array(get_lambert_arc_history(lambert_arc_ephemeris, dynamics_simulator_left.state_history))
        rel_dist_left       = {'Earth':history2array(dynamics_simulator_left.dependent_variable_history)[:,4], 'Venus':history2array(dynamics_simulator_left.dependent_variable_history)[:,5]} 
        
        
def obtain_point_mass_acceleration(relative_position,mu_parameter) :
    ## obtain point mass acceleration 
    #
    # realtive_position : body realtive to planet   ---> array(N,3)
    # mu                : gravity parameter         ---> float
    
    distance = np.sqrt(np.sum(relative_position**2,axis=1))[:,np.newaxis] #m
    acc      = (-relative_position/distance**3)*mu_parameter    # m/s^2
    return acc


lambert_sun_accelerations = dict()
for jj,case in enumerate(cases) :
    lambert_sun_accelerations[case] = obtain_point_mass_acceleration(simulations_results[case]['lambert'][:,1:4],sun_gravitational_parameter)
    
# obtain acceleration difference 

difference_numerical_analytical = dict()

for case in cases :
    difference_numerical_analytical[case] = {
    'delta_pos' : np.sqrt(np.sum((simulations_results[case]['lambert'][:,1:4]-simulations_results[case]['numerical'][:,1:4])**2,axis=1))[:,np.newaxis],
    'delta_vel' : np.sqrt(np.sum((simulations_results[case]['lambert'][:,4:7]-simulations_results[case]['numerical'][:,4:7])**2,axis=1))[:,np.newaxis],
    'delta_acc' : np.sqrt(np.sum((multi_body_acceleration[case][:,1:4]-lambert_sun_accelerations[case][:,:])**2,axis=1))[:,np.newaxis]
    }
    
######## plotting phase
# buffer 1 case

# Create traces
images_dir = 'output_images/'

Nfig = 3
fig_list = [go.Figure() for jj in range(Nfig)]

# vectors must have the same identical 1-D shape
for fig,case,title in zip(fig_list,cases,buffer_list) :
    fig = make_subplots(rows=3, cols=1)
    
    fig.add_trace(go.Scatter(x=time_stumps[case], y=difference_numerical_analytical[case]['delta_pos'][:,0],
                    mode='lines+markers'),row=1, col=1)
    fig.update_yaxes(title_text=r'$\Delta r \quad m$',nticks=4,showexponent = 'all',
        exponentformat = 'e',type="log",row=1, col=1)
    
    fig.add_trace(go.Scatter(x=time_stumps[case], y=difference_numerical_analytical[case]['delta_vel'][:,0],
                    mode='lines+markers'),
                    row=2, col=1)
    fig.update_yaxes(title_text=r'$\Delta v \quad m/s$',nticks=4,showexponent = 'all',
        exponentformat = 'e',type="log",row=2, col=1)
    
    fig.add_trace(go.Scatter(x=time_stumps[case], y=difference_numerical_analytical[case]['delta_acc'][:,0],
                    mode='lines+markers'), row=3, col=1)
    fig.update_yaxes(title_text=r'$\Delta a \quad m/s^2$ ',nticks=4,showexponent = 'all',
        exponentformat = 'e',type="log", row=3, col=1)
    
    fig.update_xaxes(title='time [days]')
    fig.add_vrect(x0=time_stumps[case][0], x1=time_stumps[case][150], 
                  annotation_text="Earth proximity", annotation_position="outside top",
                  fillcolor="green", opacity=0.25, line_width=0,row='all',col=1)
    
    
    fig.add_vrect(x0=time_stumps[case][-150], x1=time_stumps[case][-1], 
                  annotation_text="Venus proximity", annotation_position="outside top",
                  fillcolor="green", opacity=0.25, line_width=0,row='all',col=1)
    
    fig.update_layout(
        font=dict(
        family="Courier New, monospace",
        size=14,
        color="RebeccaPurple"),
        width = 1000,
        height=800,
        showlegend= False)
    file_name = images_dir + 'exercise2_{}.eps'.format(case)
    fig.write_image(file_name)
# plotting phase for middle point propagation

# state_middle_left   = history2array(dynamics_simulator_left.state_history)
# dependant_acc_left  = history2array(dynamics_simulator_left.dependent_variable_history) 
# lambert_state_left  = get_lambert_arc_history(lambert_arc_ephemeris, dynamics_simulator_left.state_history)

middle_case = ['left','right']
library     = {'left':
                 {'numerical'   :state_middle_left,  #array
                  'lambert'     :lambert_state_left, #array
                  'total_acc'   :dependant_acc_left, #array
                  'time'        :(state_middle_left[:,0]-state_middle_left[0,0])/60/60/24, #array
                  'distance'    : rel_dist_left,     #dict {'Earth'...'Venuns'}
                  },
               'right':
                  {'numerical'  :state_middle_right,
                  'lambert'     :lambert_state_right,
                  'total_acc'   :dependant_acc_right,
                  'time'        :(state_middle_right[:,0]-state_middle_right[0,0])/60/60/24,
                  'distance'    : rel_dist_right,
                  }
                  }

difference_numerical_analytical = dict()  
Nfig = 2
fig_list           = [go.Figure() for jj in range(Nfig)]   
additional_figures = [go.Figure() for jj in range(Nfig)]    
planets_list       = ['Earth','Venus']
title_spc          = ['Midlle Point to Earth','Middle point to Venus']    
   
for jj,case in enumerate(middle_case):
    
    lambert_sun_acceleration = obtain_point_mass_acceleration(library[case]['lambert'][:,1:4],sun_gravitational_parameter)
    difference_numerical_analytical[case] = {
    'delta_pos' : np.sqrt(np.sum((library[case]['lambert'][:,1:4]-library[case]['numerical'][:,1:4])**2,axis=1))[:,np.newaxis],
    'delta_vel' : np.sqrt(np.sum((library[case]['lambert'][:,4:7]-library[case]['numerical'][:,4:7])**2,axis=1))[:,np.newaxis],
    'delta_acc' : np.sqrt(np.sum((library[case]['total_acc'][:,1:4]-lambert_sun_acceleration)**2,axis=1))[:,np.newaxis]
    }
    
    #create figure
    fig=fig_list[jj]
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=library[case]['time'], y=difference_numerical_analytical[case]['delta_pos'][:,0],
    mode='lines+markers'),row=1, col=1)
    fig.update_yaxes(title_text=r'$\Delta r \quad m$',nticks=4,showexponent = 'all',
    exponentformat = 'e',type="log",row=1, col=1)
    
    
    fig.add_trace(go.Scatter(x=library[case]['time'], y=difference_numerical_analytical[case]['delta_vel'][:,0],
    mode='lines+markers'),
    row=2, col=1)
    fig.update_yaxes(title_text=r'$\Delta v \quad m/s$',nticks=4,showexponent = 'all',
    exponentformat = 'e',type="log",row=2, col=1)

    fig.add_trace(go.Scatter(x=library[case]['time'], y=difference_numerical_analytical[case]['delta_acc'][:,0],
    mode='lines+markers'), row=3, col=1)
    fig.update_yaxes(title_text=r'$\Delta a \quad m/s^2$ ',nticks=4,showexponent = 'all',
    exponentformat = 'e',type="log", row=3, col=1)

    fig.update_xaxes(title='time [days]')
    
    fig.add_vrect(x0=library[case]['time'][0], x1=library[case]['time'][150], 
                  annotation_text="Earth proximity", annotation_position="outside top",
                  fillcolor="green", opacity=0.25, line_width=0,row='all',col=1)
    
    
    fig.add_vrect(x0=library[case]['time'][-150], x1=library[case]['time'][-1], 
                  annotation_text="Venus proximity", annotation_position="outside top",
                  fillcolor="green", opacity=0.25, line_width=0,row='all',col=1)
    
    fig.update_layout(
    font=dict(
    family="Courier New, monospace",
    size=14,
    color="RebeccaPurple"),
    width = 1000,
    height=800,
    showlegend= False)

    file_name = images_dir + 'exercise2_{}.eps'.format(case)
    fig.write_image(file_name)
    fig.show()
    
    # plotting relative distances
    fig2 = additional_figures[jj]
    fig2.add_trace(go.Scatter(x=library[case]['time'], y=library[case]['distance'][planets_list[jj]],
    mode='lines+markers'))
    fig2.update_yaxes(title_text=r'$||r|| \quad [m]$',nticks=4,showexponent = 'all',
    exponentformat = 'e')
    fig2.update_xaxes(title_text='time [days]')
    
    fig2.update_layout(
    font=dict(
    family="Courier New, monospace",
    size=14,
    color="RebeccaPurple"),
    width = 1000,
    height=800,
    showlegend= False)
    
    

    file_name = images_dir + 'exercise2_relative_distance_{}.eps'.format(planets_list[jj])
    fig.write_image(file_name)

