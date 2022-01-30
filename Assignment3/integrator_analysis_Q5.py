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
import scipy.special
spice_interface.load_standard_kernels()
bodies = create_bodies( )

# Define settings for orbit only, unperturbed only
current_phase            = 1
central_body             = "Ganymede"
bodies_to_integrate      = ["JUICE"]
central_bodies           = [ central_body ]
current_phase_start_time = initial_times_per_phase[current_phase]
acceleration_models      = get_unperturbed_accelerations( central_body, bodies)

state_differences_rkf   = np.zeros((6,20))
state_differences_euler = np.zeros((6,20))

step_per_run       = 600.0
time_instances     = [step_per_run*i for i in range(80)]

Kr_rk78             = []
Kr_Euler           = []
Kr_theory_rk78      = []
Kr_theory_Euler     = []


# Perform 20 individual steps
for t in time_instances:
    
    # Compute initial time of current step
    current_start_time = current_phase_start_time + t
    
    # Compute initial state of current step
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name= "JUICE",
        observer_body_name=central_body,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time= current_start_time
    )
    

    # Define propagator settings, terminate after 300 s.
    time_step = 300.0
    termination_time     = current_start_time + time_step    
    termination_settings = propagation_setup.propagator.time_termination(
        termination_time, terminate_exactly_on_final_condition=True )
    
    dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('JUICE',central_body)]
    
    propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_integrate,
            initial_state,
            termination_settings,
            output_variables = dependent_variables_to_save)
    
    
    '''Runge Kutta '''
    # Get fixed step RKF78 integrator settings
    integrator_settings_rk78 = get_fixed_step_size_integrator_settings(current_start_time,time_step)
    # Propagate Dynamics
    dynamics_simulator_rk78  = numerical_simulation.SingleArcSimulator(bodies,
                                                                    integrator_settings_rk78,
                                                                    propagator_settings,
                                                                    print_dependent_variable_data=False)
    
    integrator_settings_euler = propagation_setup.integrator.euler(current_start_time,time_step)
    # Propagate Dynamics
    dynamics_simulator_euler  = numerical_simulation.SingleArcSimulator(bodies,
                                                                    integrator_settings_euler,
                                                                    propagator_settings,
                                                                    print_dependent_variable_data=False)
    
    # and now we really have a lot of fun
    # let's make a rotation matrices 
    
    # in theory you should not to recompute all the parameters since they 
    # should be always the same.
    # But just to be quick and leazy we will do it inside the loop every time
    
    # this parameters are the same for Euler and rk78 because they both start from the same point
    radius      = dynamics_simulator_rk78.dependent_variable_history[current_start_time][0]
    omega_peri  = dynamics_simulator_rk78.dependent_variable_history[current_start_time][3]
    Raan        = dynamics_simulator_rk78.dependent_variable_history[current_start_time][4]
    inc         = dynamics_simulator_rk78.dependent_variable_history[current_start_time][2]
    theta0      = dynamics_simulator_rk78.dependent_variable_history[current_start_time][-1]
                            
    pos = dynamics_simulator_rk78.state_history[current_start_time]
    
    # euler rot 313
    
    Raan_rot = np.array([[np.cos(Raan),    -np.sin(Raan),     0 ],
                         [np.sin(Raan),     np.cos(Raan),     0 ],
                         [           0,                0,     1 ]])
    
    inc_rot = np.array([[            1,                 0,             0],
                         [           0,     np.cos(inc),    -np.sin(inc)],
                         [           0,     np.sin(inc),     np.cos(inc)]])
    
    omega_peri_rot = np.array([[np.cos(omega_peri),    -np.sin(omega_peri),     0 ],
                               [np.sin(omega_peri),     np.cos(omega_peri),     0 ],
                               [           0,                0,                 1 ]])
    
    Rotation = Raan_rot@inc_rot@omega_peri_rot
    
    state    = Rotation@np.array([[radius*np.cos(theta0)],
                                  [radius*np.sin(theta0)],
                                  [ 0 ]]).ravel()
    
   
    mu    = bodies.get_body(central_body).gravitational_parameter
    omega = np.sqrt(mu/radius**3)
    Period = 2*np.pi/omega
    differentiated_state_rk78  = Rotation @ np.array([[radius* -omega**8*np.cos(theta0)],
                                                      [radius*  omega**8*np.sin(theta0)],
                                                      [0                              ]])/scipy.special.factorial(8)
    
    print(differentiated_state_rk78)
    differentiated_state_Euler  = Rotation @ np.array([[radius*-omega**2*np.cos(theta0)],
                                                       [radius*-omega**2*np.sin(theta0)],
                                                       [0                              ]])/scipy.special.factorial(2)
    
    
    
                                
    theoretical_Kr_rk78   = np.sum(differentiated_state_rk78**2)**0.5
    theoretical_Kr_Euler  = np.sum(differentiated_state_Euler**2)**0.5

    
    file_output_identifier = "./exercise5/time_"+str(t)
    
    LTE_rk78   = get_difference_wrt_kepler_orbit(
                                  dynamics_simulator_rk78.state_history,
                                  bodies.get_body(central_body).gravitational_parameter)[termination_time]
    
    LTE_euler = get_difference_wrt_kepler_orbit(
                                  dynamics_simulator_euler.state_history,
                                  bodies.get_body(central_body).gravitational_parameter)[termination_time]
    
    
    print(1/omega*2*np.pi)
    
    Kr_rk78_current = np.sqrt(np.sum((LTE_rk78[:3]/time_step**8)**2))
    Kr_rk78.append(Kr_rk78_current)
    
    Kr_Euler_current = np.sqrt(np.sum((LTE_euler[:3]/time_step**2)**2))
    Kr_Euler.append(Kr_Euler_current)
    
    Kr_theory_rk78.append(theoretical_Kr_rk78)
    Kr_theory_Euler.append(theoretical_Kr_Euler)
    
    
    
Kr_theory_rk78  = np.array(Kr_theory_rk78)
Kr_theory_Euler = np.array(Kr_theory_Euler)
Kr_Euler        = np.array(Kr_Euler)
Kr_rk78         = np.array(Kr_rk78)

figure,axes=plt.subplots(2,1)
axes[0].plot(time_instances/Period,Kr_Euler,c='blue',label= 'numerical') 
axes[0].plot(time_instances/Period,Kr_theory_Euler,c='green',label= 'analytical (Taylor)')
axes[0].set_title('Euler step')

axes[1].plot(time_instances/Period,Kr_rk78,c='blue')
axes[1].plot(time_instances/Period,Kr_theory_rk78,c='green') 
axes[1].set_title('RK 7/8 step')

for ax in axes : 
    ax.set_xlabel('JUICE periods')
    ax.set_ylabel(r'$||Kr||_2$')
axes[0].legend()
plt.tight_layout()
plt.show()

