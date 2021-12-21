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
from   plotly.subplots import make_subplots

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"
output_images_directory="./output_images/"
###########################################################################
# RUN CODE FOR QUESTION 5 #################################################
###########################################################################
frame_origin         = 'Sun'
bodies_to_create     = ['Sun','Mars','Earth','Venus','Moon','Jupiter']
vehicle_specific     = {'name':'Spacecraft',   #name
                        'mass':1000,           #kg
                        'radiation_ref_A':20,  #m^2
                        'radiation_coefficient':1.2}  


sun_gravitational_parameter = 1.32712440042E20 #m^3/s^2
# Create body objects
bodies                = create_simulation_bodies( bodies_to_create,frame_origin,vehicle_spec=vehicle_specific)
# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)
# this is just obtained once

# shift time according to the exercise specifications

time_buffer                 = constants.JULIAN_DAY*2

time_of_flight_with_buffer  = time_of_flight  - 2*time_buffer
departure_epoch_with_buffer = departure_epoch + time_buffer
arrival_epoch_with_buffer   = departure_epoch_with_buffer + time_of_flight_with_buffer

# target linearization error
# on position and speed
target_position_error            = 1E5
target_speed_error               = 1
            
            
# useful function 

def history2array(state_history):
    state = np.array(list(state_history.values()))
    time  = np.array(list(state_history.keys()))[:,np.newaxis]
    res   = np.hstack((time,state))
    return res

# Set arc length
number_of_arcs          = 10
arc_length              = time_of_flight_with_buffer/number_of_arcs

# some preallocations
permitted_perturbations = np.zeros((number_of_arcs,6))
permitted_rsw_frame     = np.zeros((number_of_arcs,3))


parameters = ['Dx0','Dy0','Dz0','Dvx0','Dvy0','Dvz0']
file='convergence1txt' 
with open(file, 'w') as filetowrite:
    for arc_index in range(0,number_of_arcs) :
        
        # Compute start and end time for current arc
        current_arc_initial_time = departure_epoch_with_buffer + arc_index*arc_length
        current_arc_final_time   = departure_epoch_with_buffer+ (1+arc_index)*arc_length

        # Get propagator settings for perturbed forward arc
        arc_initial_state   = lambert_arc_ephemeris.cartesian_state(current_arc_initial_time)
        propagator_settings = get_perturbed_propagator_settings(bodies, arc_initial_state, current_arc_final_time)

        # Set integrator settings
        integrator_settings = propagation_setup.integrator.runge_kutta_4( current_arc_initial_time, fixed_step_size )

        ###########################################################################
        # PROPAGATE NOMINAL TRAJECTORY AND VARIATIONAL EQUATIONS ##################
        ###########################################################################

        sensitivity_parameters = get_sensitivity_parameter_set(
            propagator_settings, bodies, target_body)
        
        variational_equations_simulator = numerical_simulation.SingleArcVariationalSimulator(
            bodies, integrator_settings, propagator_settings, sensitivity_parameters)

        state_transition_result    = variational_equations_simulator.state_transition_matrix_history
        nominal_integration_result = variational_equations_simulator.state_history  # nominal state result given the perturbations

        # Compute arc initial state before applying variations
        time                   = list(state_transition_result.keys())
        initial_epoch          = list(state_transition_result.keys())[0]
        original_initial_state = nominal_integration_result[initial_epoch]
        
        # define RSW frame matrix :
        radial      = original_initial_state[:3]/np.sqrt(np.sum(original_initial_state[:3]**2)) #normalise
        tangential  = original_initial_state[3:]/np.sqrt(np.sum(original_initial_state[3:]**2)) #normalise
        
        cross_track = np.cross(radial,tangential)
        cross_track = cross_track/np.sqrt(np.sum((cross_track)**2))#normalise
        
        along_track = np.cross(cross_track,radial)
        
        # DCM matrix
        
        DCM_inertial2RSW =  np.array([radial,along_track,cross_track])
        print(DCM_inertial2RSW)
                
        lambert_history                  = get_lambert_arc_history(lambert_arc_ephemeris, nominal_integration_result)
        nominal_integration_state        = history2array(nominal_integration_result)
        
        ###########################################################################
        # START ANALYSIS ALGORITHM FOR QUESTION 4 #################################
        ###########################################################################

        # This vector will hold the maximum permitted initial state perturbations for which the linearization 
        # is valid (for the current arc. The vector is initialized to 0, and each of its 6 entries is computed 
        # in the 6 iterations of the coming for loop (that runs over the iteration variable 'entry')
        tolerance_on_solution   = 1E-3  # to be changed
        tolerance_on_domain     = 1E-3  # 
        max_iterations          = 100
        
    
        # insert here the initial guesses for the initial
        # parameters set
        
        initial_guess = np.array([0.01,0.001,0.001,0.001,0.001,0.001])
        filetowrite.write('------------------------------\n')
        filetowrite.write('Arc number {}\n'.format(arc_index))
        filetowrite.write('------------------------------\n')
        
        print('Analysing Arch {} .....'.format(arc_index))
        # Iterate over all initial state entries
        for entry in range(6):
            print('Optimising variable {} .....'.format(parameters[entry]))
            # Define (iterative) algorithm to compute current entry of 'permitted_perturbations'
            # General structure: define an initial state perturbation (perturbed_initial_state variable),
            # compute epsilon_x (see assignment), and iterate your algorithm until convergence.
            filetowrite.write('#########################\n')
            filetowrite.write('variable                 : {}\n'.format(entry))
           
               
            iteration   = 0
            start_guess = initial_guess[entry]
            
            aa          = np.min(np.array([0.,start_guess]))
            
            if aa == 0. :
                bb          = start_guess + 1E15 # push the boundary far on the right
            if aa != 0.:
                bb          = start_guess - 1E15
                
            
            filetowrite.write('Start Guess :         {}\n'.format(start_guess))
            
            cc_perturbation        = np.array([0.,0.,0.,0.,0.,0.])
            cc_perturbation[entry] = initial_guess[entry]
            
            # starting values
            
            epsilon_r               = 0 # just a random initial state
            epsilon_v               = 0 # just a random initial state
            
            width                   = abs(bb-aa)
        
           # optimise for the position error only and check what happens to the speed
           # error in the mean while.
           # if at the end of the iteration the speed error is exceeding the limit, 
           # add an extra set of iterations to converge to the speed error that you want to
           # achieve
           
           
            while abs(epsilon_r-1E5)>=tolerance_on_solution and iteration <max_iterations and width>tolerance_on_domain :

                # Reset propagator settings with perturbed initial state
                perturbed_initial_state            = cc_perturbation + original_initial_state
                propagator_settings.initial_states = perturbed_initial_state
                dynamics_simulator                 = numerical_simulation.SingleArcSimulator(bodies,
                                                                            integrator_settings,
                                                                            propagator_settings,
                                                                            print_dependent_variable_data=False)
                
                DX_real = history2array(dynamics_simulator.state_history)[:,1:]-nominal_integration_state[:,1:]
                DX_lin  = np.empty(np.shape(DX_real))
                
                for jj,t in zip(range(len(time)),time) :
                    PHI          =  state_transition_result[t]  
                    dx           = PHI @ cc_perturbation[:,np.newaxis]
                    DX_lin[jj,:] = dx.T
                
                epsilon_r =   np.max(np.sqrt(np.sum((DX_real[:,:3] - DX_lin[:,:3])**2,axis = 1)))
                epsilon_v =   np.max(np.sqrt(np.sum((DX_real[:,3:] - DX_lin[:,3:])**2,axis = 1)))
                
                # evaluate both functions sign 
                f1 = epsilon_r - target_position_error
                f2 = epsilon_v - target_speed_error
                

                
                filetowrite.write('Iterations             : {}\n'.format(iteration))
                filetowrite.write('{} current guess     : {}\n'.format(parameters[entry],start_guess))
                filetowrite.write('Domain width           : {}\n'.format(width))
                filetowrite.write('aa                     : {}\n'.format(aa))
                filetowrite.write('bb                     : {}\n'.format(bb))
                filetowrite.write('------------------------\n')
                
                
                if f1 >0 :
                    bb = start_guess
                    start_guess = (bb+aa)/2
                    cc_perturbation[entry] = start_guess
                elif f1 <0 :
                    aa = start_guess
                    start_guess = (bb+aa)/2
                    cc_perturbation[entry] = start_guess
                
                
                width = abs(aa-bb)  
                
                iteration = iteration + 1
            
            permitted_perturbations[arc_index,entry] = start_guess
            CheckSpeedThreshold = epsilon_v - target_speed_error
                
            # if speed limit is vilated recheck the conditioon
            if  CheckSpeedThreshold >0 :
                    print('Additional Check Required : Speed limit Violated')
                    iteration   = 0
                    aa          = np.min(np.array([0.,start_guess]))
            
                    if aa == 0. :
                        bb          = start_guess  
                    if aa != 0.:
                        aa          = start_guess
                        bb          = 0
                      
                    start_guess            = (aa+bb)/2
                    cc_perturbation[entry] = start_guess
                    width                  = abs(bb-aa)
                    
                    while abs(epsilon_v-target_speed_error)>=tolerance_on_solution and iteration <max_iterations and width>tolerance_on_domain :
                        # Reset propagator settings with perturbed initial state
                        
                        perturbed_initial_state            = cc_perturbation + original_initial_state
                        propagator_settings.initial_states = perturbed_initial_state
                        dynamics_simulator = numerical_simulation.SingleArcSimulator(bodies,
                                                                                    integrator_settings,
                                                                                    propagator_settings,
                                                                                    print_dependent_variable_data=False)
                        
                        DX_real = history2array(dynamics_simulator.state_history)[:,1:]-nominal_integration_state[:,1:]
                        DX_lin  = np.empty(np.shape(DX_real))
                        
                        for jj,t in zip(range(len(time)),time) :
                            PHI          =  state_transition_result[t]  
                            dx           = PHI @ cc_perturbation[:,np.newaxis]
                            DX_lin[jj,:] = dx.T
                        
                        epsilon_r =   np.max(np.sqrt(np.sum((DX_real[:,:3] - DX_lin[:,:3])**2,axis = 1)))
                        epsilon_v =   np.max(np.sqrt(np.sum((DX_real[:,3:] - DX_lin[:,3:])**2,axis = 1)))
                        
                        # evaluate both functions sign 
                        f1 = epsilon_r - target_position_error
                        f2 = epsilon_v - target_speed_error
                        filetowrite.write('additional iterations  : {}\n'.format(f1))
                        filetowrite.write('Iterations             : {}\n'.format(iteration))
                        filetowrite.write('{}  current guess     : {}\n'.format(parameters[entry],start_guess))
                        filetowrite.write('Domain width           : {}\n'.format(width))
                        filetowrite.write('lower bound            : {}\n'.format(aa))
                        filetowrite.write('upper bound            : {}\n'.format(bb))
                        filetowrite.write('------------------------\n')
                        
                        
                        if f2 >0 :
                            bb = start_guess
                            start_guess = (bb+aa)/2
                            cc_perturbation[entry] = start_guess
                        elif f2 <0 :
                            aa = start_guess
                            start_guess = (bb+aa)/2
                            cc_perturbation[entry] = start_guess
                        
                        
                        width = abs(aa-bb)  
                        
                        iteration = iteration + 1
            permitted_perturbations[arc_index,entry] = start_guess
                        
        
            filetowrite.write('---------------------------\n')
            filetowrite.write('Summary\n')
            filetowrite.write('Parameter                  : {}\n'.format(parameters[entry]))
            filetowrite.write('Final Position Error       : {} m\n'.format(epsilon_r))
            filetowrite.write('Final Speed Error          : {} m/s\n'.format(epsilon_v))
            filetowrite.write('Iterations total           : {}\n'.format(iteration))
            filetowrite.write('Final Parameter Value      : {}\n'.format(start_guess))
            filetowrite.write('Final Domain width         : {}\n'.format(width))
    
    
   
    
    # convert frame of reference
        permitted_rsw_frame [arc_index,:3] = DCM_inertial2RSW @ permitted_perturbations[arc_index,:3][:np.newaxis]
    
          
table = open('summary_table.txt','w')
col = ['Index','$\Delta x_0 [m]$','$\Delta y_0 [m]$','$\Delta z_0 [m]$','$||\Delta r||_2$']
table.write('&'.join(col)+'\\\\' + '\n')

for N in range(number_of_arcs) :
    values = permitted_perturbations[N,:3]
    norm   = np.sqrt(np.sum(values**2))
    values_list = values.tolist()
    values_list.append(norm.tolist())
    values_list.insert(0,N)
    
    values_list = ['{:.2f}'.format(val) for val in values_list]
    table.write('&'.join(values_list)+'\\\\' + '\n')
table.close()


ii = np.arange(number_of_arcs)

fig5a = go.Figure()
fig5a.add_trace(go.Scatter(x=ii,
                           y=np.sqrt(np.sum(permitted_perturbations[:,:3]**2,axis=1)),
                                mode='lines+markers',
                                )) 
fig5a.update_yaxes(title_text=r'$||\Delta r_0||_2 \quad [m]$',showexponent = 'all',
            exponentformat = 'e')
fig5a.update_xaxes(title_text='arc index',showexponent = 'all',exponentformat = 'e',tickvals=range(9))
figName = output_images_directory +'exercise5_a.eps'

fig5a.write_image(figName)
fig5a.show()

# compute the RSW frame of reference 
# how ?
# radial component is the initial position of the spacecraft normalised.
# cross track component is in the direction of the angular momnetum vector
# The final component is obtained from the cross product of the fist two 




comp = [r'$r_R$ [m]',r'$r_S$ [m]',r'$r_W$ [m]']
fig5d=make_subplots(rows=3,cols=1)
for N in range(3) :
    
    fig5d.add_trace(go.Scatter(x=ii,
                               y=permitted_rsw_frame[:,N],
                                mode='lines+markers',
                                ),row=N+1,col=1) 
    fig5d.update_yaxes(title_text=comp[N],showexponent = 'all',
            exponentformat = 'e',row=N+1,col=1)
    fig5d.update_xaxes(title_text='arc index',showexponent = 'all',exponentformat = 'e',row=N+1,col=1)

fig5d.show()