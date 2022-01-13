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

# Load spice kernels.
spice_interface.load_standard_kernels()

# Create the bodies for the numerical simulation
bodies     = create_bodies( )
step_sizes = 25.*2**np.arange(0,4)
print(step_sizes)

# Define list of step size for integrator to take
def step_size_plotter(step_sizes:list,
                      save_results:bool = False) :
    
    """calculate analytical vs numerical difference
    
    parameters 
    ----------
    step_sizes(list)   : list all the step_sizes that you want to test 
    save_results(bool) : save tables to .dat files and save the images to file 
    """
    
    
    final_max_error_figure = dict()
    
    # Iterate over phases
    for current_phase in range( len(central_bodies_per_phase )):
        print('Loading {}'.format(phase_names[current_phase]))
        save_max_error = np.empty((len(step_sizes),2))
        # Create initial state and time
        current_phase_start_time = initial_times_per_phase[ current_phase ]
        current_phase_end_time   = current_phase_start_time + propagation_times_per_phase[ current_phase ]

        # Define current central body
        current_central_body = central_bodies_per_phase[current_phase] 
        body_to_propagate    = "JUICE"
        
        # Retrieve JUICE initial state
        initial_state = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name= "JUICE",
            observer_body_name=current_central_body,
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=current_phase_start_time
        )
        
        # Retrieve acceleration settings without perturbations
        acceleration_models = get_unperturbed_accelerations(current_central_body, bodies)
        
        termination_settings = propagation_setup.propagator.time_termination(current_phase_end_time)
        
        # Define propagator settings
        propagator_settings = propagation_setup.propagator.translational(
        [current_central_body],
        acceleration_models,
        [body_to_propagate] ,
        initial_state,
        termination_settings
        )

        
        # Iterate over step size
        for jj,step_size in enumerate(step_sizes):
            #step_size = int(step_size)
            print('Working on step size : {}s'.format(step_size))
            # Define integrator settings
            integrator_settings = get_fixed_step_size_integrator_settings(current_phase_start_time, step_size)
            # Propagate dynamics
            
            dynamics_simulator = numerical_simulation.SingleArcSimulator(bodies,
                                                                        integrator_settings,
                                                                        propagator_settings,
                                                                        print_dependent_variable_data=False)
            state_history = dynamics_simulator.state_history
            
            # Compute difference w.r.t. analytical solution to file
            central_body_gravitational_parameter = bodies.get_body( current_central_body ).gravitational_parameter
            keplerian_solution_difference = get_difference_wrt_kepler_orbit( 
                state_history, central_body_gravitational_parameter)
            
            keplerian_solution_difference_array = history2array(keplerian_solution_difference)
           
            position_error        = np.sqrt(np.sum(keplerian_solution_difference_array[:,1:4]**2,axis=1))
            
            
            max_error = np.max(position_error)
            index_max =np.argmax(position_error)
            time_max              = keplerian_solution_difference_array[index_max,0]
            
            
            
            
            
            
            save_max_error[jj,1]    = position_error[-1]
            save_max_error[jj,0]    = step_size
            
            if save_results :
                #Write results to files
                folder = "exercise1/"
                file_output_identifier = folder + "Q1_step_size_" + str(int(step_size)) + "_phase_index" + str(phase_names[current_phase])   
                write_propagation_results_and_analytical_difference_to_file( 
                    dynamics_simulator, file_output_identifier, bodies.get_body( current_central_body ).gravitational_parameter)
        
        final_max_error_figure[phase_names[current_phase]] = save_max_error
            
            
        if save_results :  
            search_dir   = "./SimulationOutput/exercise1"
            output_image = './SimulationOutput/exercise1/output_images/' + phase_names[current_phase] + '.eps'
            table_name   = './SimulationOutput/exercise1/output_images/' + phase_names[current_phase] + '_table.txt'
            ax = multiplot(search_dir,
                        ['index'+phase_names[current_phase],"keplerian"],
                        r'$\epsilon$',
                        output_image,
                        table_name)
            
            
    
    return final_max_error_figure

# first part of the exercise
#step_size_plotter(step_sizes,save_results=True)

#second part of the exercise
step_sizes    = 10.**np.linspace(1.0,2.3,60)

#step_sizes = np.array([1,5,10,20,25.0006,25.,30,40,50,85,100,150,200,400,800,1000])
final_solution = step_size_plotter(step_sizes,save_results=False)

fig, ax       = plt.subplots(1,2)    

for jj,phase_name in enumerate(phase_names):
        
    sol  = final_solution[phase_name]
    
    ax[jj].plot(sol[:,0],sol[:,1])            
    ax[jj].set_xlabel('time_step [s]')
    ax[jj].set_ylabel(r'$\epsilon_{max} [m]$')
    ax[jj].set_yscale('log')
    ax[jj].set_title(phase_name)
    
fig.tight_layout()
plt.show()
