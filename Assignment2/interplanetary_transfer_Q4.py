''' 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

from numpy import histogram_bin_edges
from interplanetary_transfer_helper_functions import *
from   interplanetary_transfer_helper_functions import *
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from   plotly.subplots import make_subplots

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"
output_images_directory = './output_images/'
###########################################################################
# RUN CODE FOR QUESTION 4a #################################################
###########################################################################
# initial settings
frame_origin         = 'Sun'
bodies_to_create     = ['Sun','Mars','Earth','Venus','Moon','Jupiter']
vehicle_specific     = {'name':'Spacecraft',   #name
                        'mass':1000,           #kg
                        'radiation_ref_A':20,  #m^2
                        'radiation_coefficient':1.2}  

rsw_acceleration_magnitude = [0, 0, 0]

sun_gravitational_parameter = 1.32712440042E20 #m^3/s^2

# Create body objects
bodies = create_simulation_bodies( bodies_to_create,frame_origin,vehicle_spec=vehicle_specific)
# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)
# this is just obtained once

# shift time according to the exercise specifications

time_buffer                 = constants.JULIAN_DAY*2

time_of_flight_with_buffer  = time_of_flight  - 2*time_buffer
departure_epoch_with_buffer = departure_epoch + time_buffer
arrival_epoch_with_buffer   = departure_epoch_with_buffer + time_of_flight_with_buffer

# useful function 

def history2array(state_history):
    state = np.array(list(state_history.values()))
    time  = np.array(list(state_history.keys()))[:,np.newaxis]
    res   = np.hstack((time,state))
    return res

# Solve for state transition matrix on current arc
variational_equations_solver = propagate_variational_equations(
    departure_epoch_with_buffer,
    arrival_epoch_with_buffer,
    bodies,
    lambert_arc_ephemeris,
    use_rsw_acceleration = True) # this true is only for analytical purposes. You don't
                                 # need to input anything 

sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
state_history              = variational_equations_solver.state_history  #solution of the perturbed problem
lambert_history            = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

# NO TIME IN THIS MATRIX
state_difference     = history2array(state_history)[:,1:] - history2array(lambert_history)[:,1:]
time                 = history2array(state_history)[:,0]
save_error_over_time = {'state' : state_difference,
                        'time'  : time/constants.JULIAN_DAY - time[0]/constants.JULIAN_DAY}


final_position_error     = state_difference[-1,:3] # this has shape (3,)
final_time               = time[-1]
final_sensitivity_matrix = sensitivity_matrix_history[final_time]

# decompose sensitivity matrix in submatrices
# S(t0,t1) =  S11 --> acting on position 
#             S21 --> acting on speed

S11 = final_sensitivity_matrix[:3,:]
S21 = final_sensitivity_matrix[3:,:]

# Compute low-thrust RSW acceleration to meet required final position
rsw_acceleration_magnitude = np.linalg.inv(S11) @ -final_position_error[:,np.newaxis] #change shape for matrix multiplication

# Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
# input to the propagate_trajectory function
dynamics_simulator_corrected = propagate_trajectory(
        departure_epoch_with_buffer,
        arrival_epoch_with_buffer,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations= True,
        initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
        use_rsw_acceleration = True,
        rsw_acceleration_magnitude = rsw_acceleration_magnitude)


corrected_state = history2array(dynamics_simulator_corrected.state_history)[:,1:] - history2array(lambert_history)[:,1:]
save_error_over_time_corrected= {'state' : history2array(dynamics_simulator_corrected.state_history)[:,1:] - history2array(lambert_history)[:,1:],
                                 'time'  : time/constants.JULIAN_DAY - time[0]/constants.JULIAN_DAY } #days

initial_time  = save_error_over_time_corrected['time'][0] #days


fig4a = go.Figure()
fig4a.add_trace(go.Scatter(x=save_error_over_time_corrected['time'],
                    y=np.sqrt(np.sum(save_error_over_time_corrected['state'][:,:3]**2,axis=1)),
                    mode='lines+markers',name='Low-Thrust correction') )
fig4a.add_trace(go.Scatter(x=save_error_over_time['time'],
                    y=np.sqrt(np.sum(save_error_over_time['state'][:,:3]**2,axis=1)),
                    mode='lines+markers',name='no Low-Thrust correction') )

fig4a.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
exponentformat = 'e',showline=True, linewidth=2, linecolor='black')

fig4a.update_xaxes(title_text='time [days]',showexponent = 'all',
exponentformat = 'e',showline=True, linewidth=2, linecolor='black')

fig4a.update_layout(
font=dict(
family="Courier New, monospace",
size=14,
color="RebeccaPurple"),
width = 1000,
height=800,
showlegend= True)

figName = output_images_directory +'exercise4_single_arc.eps'
fig4a.write_image(figName)


###########################################################################
# RUN CODE FOR QUESTION 4e ################################################
###########################################################################

# divide the trajectory into 2 separate arcs of time 
# compute optimal p2


dynamics_simulator_no_thrust_full_arc = propagate_trajectory(
        departure_epoch_with_buffer,
        arrival_epoch_with_buffer,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations= True,
        initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
        use_rsw_acceleration = False)


full_arc_history = dynamics_simulator_no_thrust_full_arc.state_history
lambert_full_arc_history = get_lambert_arc_history(lambert_arc_ephemeris,full_arc_history)
time             = list(full_arc_history.keys())
final_state_position_error = full_arc_history[time[-1]][:3] - lambert_full_arc_history[time[-1]][:3]

# Time to compute variational equations

arc_lenght                              = time_of_flight_with_buffer/2
arc_name = ['arc1','arc2']
save_variable = {'arc1':[],'arc2': []}

for (arc_number,name) in zip(range(2),arc_name):
        
    # sequential propapagion 
    arc_initial_time = departure_epoch_with_buffer + arc_number*arc_lenght
    arc_final_time   = departure_epoch_with_buffer + (arc_number+1)*arc_lenght

    variational_equations_solver = propagate_variational_equations(
        arc_initial_time,
        arc_final_time,
        bodies,
        lambert_arc_ephemeris,
        use_rsw_acceleration = True) 
    
    sensitivity_matrix_history      = variational_equations_solver.sensitivity_matrix_history
    state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history
    time                            = list(sensitivity_matrix_history.keys())
    final_time                      = time[-1]
    final_state_transition_matrix   = state_transition_matrix_history[final_time]
    final_sensitivity_matrix        = sensitivity_matrix_history[final_time]
    
    save_variable[name]             = {'sensitivity' :final_sensitivity_matrix,
                                       'state_transition' :final_state_transition_matrix}
    
 
    
# Now the computation follows from the starting guess for p1 in the previous case and than 
# correct it halfway
p1           = rsw_acceleration_magnitude
S1           = save_variable['arc1']['sensitivity'][:,:]
S2_rp        = save_variable['arc2']['sensitivity'][:3,:]
phi2_r_only  = save_variable['arc2']['state_transition'][:3,:]



p2      = np.linalg.inv(S2_rp)@(-final_state_position_error[:,np.newaxis] - phi2_r_only @ S1 @ p1)

# propagate with new settings 
thrust     = [p1,p2]
print(thrust[0])
print(thrust[1])
correction = np.zeros((6,))
save_error = []

for (arc_number,name) in zip(range(2),arc_name):
     
    # sequential propapagion 
    arc_initial_time = departure_epoch_with_buffer + arc_number*arc_lenght
    arc_final_time   = departure_epoch_with_buffer + (arc_number+1)*arc_lenght


    dynamics_simulator_corrected = propagate_trajectory(
            arc_initial_time,
            arc_final_time,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations= True,
            initial_state_correction=correction,
            use_rsw_acceleration = True,
            rsw_acceleration_magnitude = np.squeeze(thrust[arc_number]))
    
    state_history       = dynamics_simulator_corrected.state_history
    lambert_arc_history = get_lambert_arc_history(lambert_arc_ephemeris,state_history)
    
    time = list(state_history.keys())
    final_state_error = state_history[time[-1]] - lambert_arc_history[time[-1]]
    
    correction = final_state_error
    state_error_array = history2array(state_history)-history2array(lambert_arc_history)
    position_error    = np.sqrt(np.sum(state_error_array[:,1:4]**2,axis=1))
    book = {'time':time,'error':position_error,'state_error':state_error_array}
    save_error.append(book)


initial_time = save_error[0]['time'][0]/constants.JULIAN_DAY
fig4c = go.Figure()
fig4c.add_trace(go.Scatter(x=np.asarray(save_error[0]['time'],np.float64)/constants.JULIAN_DAY-initial_time,
                    y=save_error[0]['error'],
                    mode='lines+markers',name='Arc1'))
fig4c.add_trace(go.Scatter(x=np.asarray(save_error[1]['time'],np.float64)/constants.JULIAN_DAY-initial_time,
                    y=save_error[1]['error'],
                    mode='lines+markers',name='Arc1'))

fig4c.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
exponentformat = 'e',showline=True, linewidth=2, linecolor='black')

fig4c.update_xaxes(title_text='time [days]',showexponent = 'all',
exponentformat = 'e',showline=True, linewidth=2, linecolor='black')

fig4c.update_layout(
font=dict(
family="Courier New, monospace",
size=14,
color="RebeccaPurple"),
width = 1000,
height=800,
showlegend= True)


figName = output_images_directory +'exercise4_two_arcs_1.eps'
fig4c.write_image(figName)

# final correction
# god make it work please

# the idea : the mid position error is so high that the 
#            the linearisation assumption maybe not valid 
#            anymore. Solev by back correcting from the final given position
#            of the lambert arc

mid_point_position_error   =  save_error[0]['state_error'][-1,1:4] # state error
mid_point_state_error      =  save_error[0]['state_error'][-1,1:]
final_state_position_error =  save_error[1]['state_error'][-1,1:4]

# so now back correct
#p2 = np.linalg.inv(S2_rp)@(-final_state_position_error+mid_point_position_error)[:,np.newaxis]

p2      = p2 + np.linalg.inv(S2_rp)@(-final_state_position_error[:,np.newaxis] - phi2_r_only @ S1 @ p1)
correction = mid_point_state_error
#correction = np.zeros((6,))
thrust = p2
arc_initial_time = departure_epoch_with_buffer + 1*arc_lenght
arc_final_time   = departure_epoch_with_buffer + (2)*arc_lenght

# new arc two propagation

dynamics_simulator_corrected = propagate_trajectory(
            arc_initial_time,
            arc_final_time,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations= True,
            initial_state_correction=correction,
            use_rsw_acceleration = True,
            rsw_acceleration_magnitude = thrust)


state_history       = dynamics_simulator_corrected.state_history
lambert_arc_history = get_lambert_arc_history(lambert_arc_ephemeris,state_history)
state_error_array   = history2array(state_history)-history2array(lambert_arc_history)
position_error      = np.sqrt(np.sum(state_error_array[:,1:4]**2,axis=1))
book                = {'time':time,'error':position_error,'state_error':state_error_array}
save_error[1]['error']       = position_error
save_error[1]['state_error'] = state_error_array


fig4e = go.Figure()
fig4e.add_trace(go.Scatter(x=np.asarray(save_error[0]['time'],np.float64)/constants.JULIAN_DAY-initial_time,
                    y=save_error[0]['error'],
                    mode='lines+markers',name='Arc1'))
fig4e.add_trace(go.Scatter(x=np.asarray(save_error[1]['time'],np.float64)/constants.JULIAN_DAY-initial_time,
                    y=save_error[1]['error'],
                    mode='lines+markers',name='Arc1'))

fig4e.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
exponentformat = 'e',showline=True, linewidth=2, linecolor='black')

fig4e.update_xaxes(title_text='time [days]',showexponent = 'all',
exponentformat = 'e',showline=True, linewidth=2, linecolor='black')

fig4e.update_layout(
font=dict(
family="Courier New, monospace",
size=14,
color="RebeccaPurple"),
width = 1000,
height=800,
showlegend= True)


fig4e.show()