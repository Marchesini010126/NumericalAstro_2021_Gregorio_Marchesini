''' 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from   plotly.subplots import make_subplots

from interplanetary_transfer_helper_functions import *

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"
images_dir       =  "./output_images"
###########################################################################
# RUN CODE FOR QUESTION 3 #################################################
###########################################################################


# DECISON TO TAKE :

activate_recursive_correction  = 0# 0/1  not_active/active



# initial settings
frame_origin         = 'Sun'
bodies_to_create     = ['Sun','Mars','Earth','Venus','Moon','Jupiter']
vehicle_specific     = {'name':'Spacecraft',   #name
                        'mass':1000,           #kg
                        'radiation_ref_A':20,  #m^2
                        'radiation_coefficient':1.2}   

sun_gravitational_parameter = 1.32712440042E20 #m^3/s^2

# Create body objects
bodies = create_simulation_bodies(bodies_to_create,frame_origin,vehicle_spec=vehicle_specific)
# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)
# this is just obtained once

# shift time according to the exercise specifications

time_buffer     = constants.JULIAN_DAY*2

departure_epoch = departure_epoch + time_buffer
arrival_epoch   = departure_epoch + time_of_flight - 2*time_buffer
time_of_flight  = time_of_flight  - 2*time_buffer

##############################################################
# Compute number of arcs and arc length
number_of_arcs = 10
arc_length     = time_of_flight/number_of_arcs
##############################################################
def history2array(state_history):
    state = np.array(list(state_history.values()))
    time  = np.array(list(state_history.keys())).reshape((-1,1))
    res   = np.hstack((time,state))
    return res

# preallocations

save_hopping_error           = dict()
save_corrected_hopping_error = dict()
total_dv_required            = 0

output_file = open('./iterations.txt','w')
output_file.write('ArcIndex & Iterations & Position Error\n')  
# Compute relevant parameters (dynamics, state transition matrix, Delta V) for each arc
for arc_index in range(number_of_arcs):

    # Compute initial and final time for arc
    current_arc_initial_time = departure_epoch   +  arc_index*arc_length
    current_arc_final_time   = departure_epoch   + (arc_index+1)*arc_length

    ###########################################################################
    # RUN CODE FOR QUESTION 3a ################################################
    ###########################################################################

    # Propagate dynamics on current arc (use propagate_trajecory function)
    dynamics_simulator = propagate_trajectory( current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                     use_perturbations = True)
    write_propagation_results_to_file(
           dynamics_simulator, lambert_arc_ephemeris, 'Q3a_arc_'+ str(arc_index),output_directory)
    ###########################################################################
    # RUN CODE FOR QUESTION 3c/d/e ############################################
    ###########################################################################
    # Note: for question 3e, part of the code below will be put into a loop
    # for the requested iterations

    # Solve for state transition matrix on current arc
    # variational equations are propagated using the linearized 
    # dynamics that follows the Lambert formulation
    
    # so the state transitionatrix is the same in the same arc even if you
    # have recursion
    max_iter = 200
    tol      = 1  # m 
    counter  = 0
    position_error_norm = 10
    initial_state_correction = np.array([0.,0.,0.,0.,0.,0.]) 
    save_iterations = []
    
    if activate_recursive_correction :
        while counter<max_iter and position_error_norm > tol :
            
            # the new linearisazion orbit should be iteratively updated
            variational_equations_solver = propagate_variational_equations(current_arc_initial_time,
                                                                    current_arc_final_time, bodies,
                                                                    lambert_arc_ephemeris,
                                                                    initial_state_correction=initial_state_correction)

            state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history
            state_history                   = variational_equations_solver.state_history
            
            # this should be propagated only once
            lambert_history                 = get_lambert_arc_history(lambert_arc_ephemeris, state_history)
        
            # Get final state transition matrix (and its inverse)
            # the matrix is a 6x6 thast can be decomposed into
            # four sub matrices 3x3
            
            final_epoch                   = list(state_transition_matrix_history.keys())[-1]
            final_state_transition_matrix = state_transition_matrix_history[final_epoch]
            
            Phi_11 = final_state_transition_matrix[:3,:3]
            Phi_12 = final_state_transition_matrix[:3,3:]
            
            Phi_21 = final_state_transition_matrix[3:,:3]
            Phi_22 = final_state_transition_matrix[3:,3:]

            # Retrieve final state deviation
            final_state_deviation    = (state_history[final_epoch] - lambert_history[final_epoch])[:,np.newaxis]
            final_position_deviation = final_state_deviation[:3,0]
        
            # Compute required velocity change at beginning of arc to meet required final state
            # change only the speed at the beginnning to have the position equal to the final 
            # position of the lambert arc
            
            initial_speed_correction = np.linalg.inv(Phi_12) @ -final_position_deviation
            # I need to produce a deviation that is the opposite of the deviaton
            # due to the force model difference. So at the end the two effects
            # will cancel out
            
            initial_state_correction += np.concatenate((np.zeros((3,)),initial_speed_correction),axis=0)
            # Propagate with correction to initial state (use propagate_trajecory function),
            # and its optional initial_state_correction input
            dynamics_simulator_corrected = propagate_trajectory( current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                                                                 use_perturbations = True,initial_state_correction=initial_state_correction)
            
            
            save_corrected_hopping_error[arc_index]   = {'state' :history2array(dynamics_simulator_corrected.state_history)[:,1:] - history2array(lambert_history)[:,1:],
                                                         'time'  : history2array(dynamics_simulator_corrected.state_history)[:,0]}
            
            final_position_error  =  save_corrected_hopping_error[arc_index]['state'][-1,:3]
            final_speed_error     =  save_corrected_hopping_error[arc_index]['state'][-1,3:]
            position_error_norm   =  np.sqrt(np.sum(final_position_error**2))
            counter               =  counter +1
            
        total_dv_required += (initial_speed_correction - final_speed_error)
        print('the total DV correction is equal to :')
        print(total_dv_required)
            
             
        
        line_table = [arc_index,counter,position_error_norm]
        line_table = [str(entry) for entry in line_table]
        output_file.write('&'.join(line_table)+'\\ \n')         
        save_iterations.append(counter) 
        
    else  :
            variational_equations_solver = propagate_variational_equations(current_arc_initial_time,
                                                                    current_arc_final_time, bodies,
                                                                    lambert_arc_ephemeris)

            state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history
            state_history                   = variational_equations_solver.state_history
            lambert_history                 = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

            save_hopping_error[arc_index]   = {'state' :history2array(dynamics_simulator.state_history)[:,1:] - history2array(lambert_history)[:,1:],
                                            'time'  : history2array(dynamics_simulator.state_history)[:,0]}
        
            # Get final state transition matrix (and its inverse)
            # the matrix is a 6x6 thast can be decomposed into
            # four sub matrices 3x3
            
            final_epoch                   = list(state_transition_matrix_history.keys())[-1]
            final_state_transition_matrix = state_transition_matrix_history[final_epoch]
            
            Phi_11 = final_state_transition_matrix[:3,:3]
            Phi_12 = final_state_transition_matrix[:3,3:]
            
            Phi_21 = final_state_transition_matrix[3:,:3]
            Phi_22 = final_state_transition_matrix[3:,3:]

            # Retrieve final state deviation
            final_state_deviation = (state_history[final_epoch] - lambert_history[final_epoch])[:,np.newaxis]

            
            # Compute required velocity change at beginning of arc to meet required final state
            # change only the speed at the beginnning to have the position equal to the final 
            # position of the lambert arc
            
            final_position_deviation = final_state_deviation[:3,0]
            initial_speed_correction = np.linalg.inv(Phi_12) @ -final_position_deviation
            # I need to produce a deviation that is the opposite of the deviaton
            # due to the force model difference. So at the end the two effects
            # will cancel out
            
            initial_state_correction = np.concatenate((np.zeros((3,)),initial_speed_correction),axis=0)
            print(initial_state_correction)
            # Propagate with correction to initial state (use propagate_trajecory function),
            # and its optional initial_state_correction input
            dynamics_simulator_corrected = propagate_trajectory( current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                            use_perturbations = True,initial_state_correction=initial_state_correction)
            lambert_history                    = get_lambert_arc_history(lambert_arc_ephemeris, dynamics_simulator_corrected.state_history)
            save_corrected_hopping_error[arc_index]   = {'state' :history2array(dynamics_simulator_corrected.state_history)[:,1:] - history2array(lambert_history)[:,1:],
                                                        'time'  : history2array(dynamics_simulator_corrected.state_history)[:,0]}
            
            #initial and final dv correction in order to remain into the 
            # lambert arc linearization
            
            
                                 # initial speed correction   # final speed correction burn
            total_dv_required += initial_speed_correction - save_corrected_hopping_error[arc_index]['state'][-1,3:]
            print('the total DV correction is equal to :')
            print(total_dv_required)


# plot 3a part
# uncorrected hopping error
if not activate_recursive_correction :
    fig3a = go.Figure()
    initial_time  = save_hopping_error[0]['time'][0]/constants.JULIAN_DAY

    for jj in range(number_of_arcs) :
       fig3a.add_trace(go.Scatter(x=save_hopping_error[jj]['time']/constants.JULIAN_DAY-initial_time,
                                y=np.sqrt(np.sum(save_hopping_error[jj]['state'][:,:3]**2,axis=1)),
                                mode='lines+markers',
                                name="arc_number : {}".format(jj))) 

    fig3a.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
            exponentformat = 'e',type="log", range=[-1,9])
    fig3a.update_xaxes(title_text='time [days]',showexponent = 'all',
            exponentformat = 'e')

    fig3a.update_layout(
            font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"),
            width = 1000,
            height=800,
            showlegend= True)

    file_name = images_dir + '/exercise3a.eps'
    fig3a.show()
    fig3a.write_image(file_name)

    # plot 3a part
    # uncorrected hopping error

    fig3c         = go.Figure()
    initial_time  = save_corrected_hopping_error[0]['time'][0]/constants.JULIAN_DAY

    for jj in range(number_of_arcs) :
        fig3c.add_trace(go.Scatter(x=save_corrected_hopping_error[jj]['time']/constants.JULIAN_DAY-initial_time,
                                y=np.sqrt(np.sum(save_corrected_hopping_error[jj]['state'][:,:3]**2,axis=1)),
                                mode='lines+markers',
                                name="arc_number : {}".format(jj))) 

    fig3c.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
            exponentformat = 'e',type="log", range=[-1,9])
    fig3c.update_xaxes(title_text='time [days]',showexponent = 'all',
            exponentformat = 'e')

    fig3c.update_layout(
            font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"),
            width = 1000,
            height=800,
            showlegend= True)
    fig3c.add_annotation(
        x=15.5,
        y=6E5,
        xref="x",
        yref="y",
        text="Error 2.5E5",
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=100,
        ay=-250,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
        )
    
    fig3c.add_annotation(
        x=147.5,
        y=3.17E4,
        xref="x",
        yref="y",
        text="Error 3.17E4",
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
        )
    
    file_name = images_dir + '/exercise3c.eps'
    fig3c.write_image(file_name)
    fig3c.show()

else :
    fig3e         = go.Figure()
    initial_time  = save_corrected_hopping_error[0]['time'][0]/constants.JULIAN_DAY

    for jj in range(number_of_arcs) :
        fig3e.add_trace(go.Scatter(x=save_corrected_hopping_error[jj]['time']/constants.JULIAN_DAY-initial_time,
                                y=np.sqrt(np.sum(save_corrected_hopping_error[jj]['state'][:,:3]**2,axis=1)),
                                mode='lines+markers',
                                name="arc_number : {}".format(jj))) 

    fig3e.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
            exponentformat = 'e',type="log", range=[-1,9])
    fig3e.update_xaxes(title_text='time [days]',showexponent = 'all',
            exponentformat = 'e')

    fig3e.update_layout(
            font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"),
            width = 1000,
            height=800,
            showlegend= True)
    
    
    file_name = images_dir + '/exercise3e.eps'
    fig3e.write_image(file_name)
    fig3e.show()

    output_file.close()