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


save_error_over_time_corrected['time']
save_error_over_time_corrected['state']

print(arrival_epoch_with_buffer)
for ii in range(len(history2array(dynamics_simulator_corrected.state_history)[-1,1:])):
  print('{:,.15f}'.format(history2array(dynamics_simulator_corrected.state_history)[-1,1+ii]))

fig4a = go.Figure()
fig4a.add_trace(go.Scatter(x=save_error_over_time_corrected['time'],
                    y=np.sqrt(np.sum(save_error_over_time_corrected['state'][:,:3]**2,axis=1)),
                    mode='lines+markers',name='Low-Thrust correction') )
fig4a.add_trace(go.Scatter(x=save_error_over_time['time'],
                    y=np.sqrt(np.sum(save_error_over_time['state'][:,:3]**2,axis=1)),
                    mode='lines+markers',name='no Low-Thrust correction') )

fig4a.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
exponentformat = 'e',type="log", range=[-1,11])

fig4a.update_xaxes(title_text='time [days]',showexponent = 'all',
exponentformat = 'e')

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
                    mode='lines+markers',name='Arc2'))

fig4c.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
exponentformat = 'e',type='log')

fig4c.update_xaxes(title_text='time [days]',showexponent = 'all',
exponentformat = 'e')

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

###########################################################################
# RUN CODE FOR QUESTION 4e ################################################
###########################################################################

# step by step solution
# (some passages are identical to previous passages, but I 
#  repeat them for clarity)

# step 1 : propagate perturbed orbit with p1 thrust until mid point
# step 2 : stop at mid point and recompute the state error propagatetd vs Lambert
# step 3 : propagate perturbed from the new mid point position and propagate
#          variational equations 
# step 4 : apply a new thrust correction that will correct the final error 
#          computed between the solution at step 3  and Lambert solution
# step 5 : if some error is still present it is possible to 
#          compensate for it by iterative process on p2



# step 1 : propagate perturbed orbit with p1 thrust 

# from start to midpoint
arc1_initial_time = departure_epoch_with_buffer 
arc1_final_time   = departure_epoch_with_buffer + arc_lenght

# uncorrected perturbe orbit
dynamics_simulator = propagate_trajectory(
            arc1_initial_time,
            arc1_final_time,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations= True,
            use_rsw_acceleration = True,
            rsw_acceleration_magnitude = p1)

# step 2 : stop at mid point and recompute the state error propagatetd vs Lambert

arc1_state_history     = dynamics_simulator.state_history
time_arc1              = list(arc1_state_history.keys())
lambert_arc1_history   = get_lambert_arc_history(lambert_arc_ephemeris,arc1_state_history)
midpoint_correction    = arc1_state_history[time_arc1[-1]]-lambert_arc1_history[time_arc1[-1]]

#step 3 : propagate perturbed from the new mid point position and propagate
#          variational equations 

arc2_initial_time = departure_epoch_with_buffer + arc_lenght
arc2_final_time   = departure_epoch_with_buffer + 2*arc_lenght

# solves both the perturbed state and the variational equations
variational_equations_solver = propagate_variational_equations(
        arc2_initial_time,
        arc2_final_time,
        bodies,
        lambert_arc_ephemeris,
        initial_state_correction = midpoint_correction,
        use_rsw_acceleration = True) 

arc2_state_history     = variational_equations_solver.state_history
time_arc2              = list(arc2_state_history.keys())
print('{:,.15f}'.format(time_arc2[-1]))
lambert_arc2_history   = get_lambert_arc_history(lambert_arc_ephemeris,arc2_state_history)

final_error             = arc2_state_history[time_arc2[-1]]-lambert_arc2_history[time_arc2[-1]]
final_position_error    = final_error[:3]
St1t2                   = variational_equations_solver.sensitivity_matrix_history[time_arc2[-1]]


# step 4/5 : apply a new thrust correction that will correct the final error 
#            computed between the solution at step 3 and Lambert solution. Also implement 
#            this passage iteratively so that the final solution will converge

# compute the new thrust
St1t2_rp   = St1t2[:3,:]
p2         = -np.linalg.inv(St1t2_rp)@final_error[:3][:,np.newaxis]

counter  = 0
max_iter = 30
tol      = 10 #m

while np.linalg.norm(final_position_error) > tol and counter <max_iter :

    dynamics_simulator = propagate_trajectory(
                arc2_initial_time,
                arc2_final_time,
                bodies,
                lambert_arc_ephemeris,
                initial_state_correction = midpoint_correction,
                use_perturbations= True,
                use_rsw_acceleration = True,
                rsw_acceleration_magnitude = p2)

    arc2_corrected_state_history     = dynamics_simulator.state_history
    time_arc2                        = list(arc2_state_history.keys())
    lambert_arc2_history             = get_lambert_arc_history(lambert_arc_ephemeris,arc2_corrected_state_history)

    final_error             = arc2_corrected_state_history[time_arc2[-1]]-lambert_arc2_history[time_arc2[-1]]
    final_position_error    = final_error[:3]
    
    p2         = p2 + -np.linalg.inv(St1t2_rp)@final_error[:3][:,np.newaxis]
    
    counter = counter + 1


# Conversion to array for semplification

position_difference_arc1 = history2array(arc1_state_history)[:,:3] - history2array(lambert_arc1_history)[:,:3]
position_difference_arc2 = history2array(arc2_corrected_state_history)[:,:3] - history2array(lambert_arc2_history)[:,:3]

position_difference_norm_arc1= np.sqrt(np.sum(position_difference_arc1**2,axis=1))
position_difference_norm_arc2= np.sqrt(np.sum(position_difference_arc2**2,axis=1))

time_arc1 = np.array(time_arc1)/constants.JULIAN_DAY
intial_time = time_arc1[-1]
  
time_arc1 = time_arc1 - initial_time
time_arc2 = np.array(time_arc2)/constants.JULIAN_DAY - initial_time



fig4e = go.Figure()
fig4e.add_trace(go.Scatter(x=time_arc1,
                           y=position_difference_norm_arc1,
                    mode='lines+markers',name='Arc1'))

fig4e.add_trace(go.Scatter(x=time_arc2,
                    y=position_difference_norm_arc2,
                    mode='lines+markers',name='Arc2'))

fig4e.update_yaxes(title_text=r'$||\Delta r||_2 \quad [m]$',showexponent = 'all',
exponentformat = 'e',type='log')

fig4e.update_xaxes(title_text='time [days]',showexponent = 'all',
exponentformat = 'e')

fig4e.update_layout(
font=dict(
family="Courier New, monospace",
size=14,
color="RebeccaPurple"),
width = 1000,
height=800,
showlegend= True)

figName = output_images_directory +'exercise4e.eps'
fig4e.write_image(figName)
fig4e.show()