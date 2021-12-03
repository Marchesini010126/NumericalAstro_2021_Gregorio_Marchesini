###########################################################################
#
# # Numerical Astrodynamics 2021/2022
#
# # Assignment 1 - Propagation Settings
#
###########################################################################
#%%


''' 
Copyright (c) 2010-2020, Delft University of Technology
All rights reserved

This file is part of Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

## Some basic plotting specifications 
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

# Main TuDat imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation # KC: newly added by me
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.util import result2array



# # student number: 1244779 --> 1244ABC
A = 8
B = 2
C = 4

simulation_start_epoch = 33.15 * constants.JULIAN_YEAR + A * 7.0 * constants.JULIAN_DAY + B * constants.JULIAN_DAY + C * constants.JULIAN_DAY / 24.0
simulation_end_epoch = simulation_start_epoch + 344.0 * constants.JULIAN_DAY / 24.0

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Load spice kernels.
spice.load_standard_kernels() # load the kernel?

# Create settings for celestial bodies
bodies_to_create         = ['Ganymede','Jupiter']         # this must have a list of all the planets to create
global_frame_origin      = 'Jupiter'        # this is the origin of the refernce system
global_frame_orientation = 'ECLIPJ2000'  # orientation of the reference system
body_settings            = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation) # body settings taken from SPICE.


# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################


    


# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['Ganymede']
central_bodies      = ['Jupiter']   # body around which the propapagtion is taken

# Define accelerations acting on Ganymede.
## CASE1 : unperturbed orbit
acceleration_settings_on_ganymede_unperturbed = dict(
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()]  # create a list of possible accelerations you want. In this case you have one single body and his atmosphere
)
## CASE2 : perturbed orbit
acceleration_settings_on_ganymede_perturbed = dict(
    Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2,0,2,2)]  # create a list of possible accelerations you want. In this case you have one single body and his atmosphere
)


# Create global accelerations dictionary.
acceleration_settings_unperturbed = {'Ganymede': acceleration_settings_on_ganymede_unperturbed}
# Create global accelerations dictionary.
acceleration_settings_perturbed = {'Ganymede': acceleration_settings_on_ganymede_perturbed}

# Create acceleration models.
acceleration_models_unperturbed = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_unperturbed, bodies_to_propagate, central_bodies)

acceleration_models_perturbed = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_perturbed, bodies_to_propagate, central_bodies)
###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define initial state.
system_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='Ganymede',
    observer_body_name='Jupiter',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time= simulation_start_epoch )

# Define required outputs
dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('Ganymede','Jupiter')]
# note here you made the error of not creating a list. you should create  alist of outputs.
# Read carefully the errors since they are really explicits when you know the type of each variable.


# Create propagation settings.
termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )
propagator_settings_unperturbed = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models_unperturbed,
    bodies_to_propagate,
    system_initial_state,
    termination_settings,
    output_variables = dependent_variables_to_save
)

propagator_settings_perturbed = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models_perturbed,
    bodies_to_propagate,
    system_initial_state,
    termination_settings,
    output_variables = dependent_variables_to_save
)
    
    
# Create numerical integrator settings.
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch,
    fixed_step_size
)

###########################################################################
# PROPAGATE ORBIT #########################################################
###########################################################################

# Create simulation object and propagate dynamics.
dynamics_simulator_unperturbed= numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings_unperturbed )
# Create simulation object and propagate dynamics.
dynamics_simulator_perturbed= numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings_perturbed )


simulation_result_unperturbed = dynamics_simulator_unperturbed.state_history
dependent_variables_unperturbed = dynamics_simulator_unperturbed.dependent_variable_history  # here you define a ditionary

simulation_result_perturbed = dynamics_simulator_perturbed.state_history
dependent_variables_perturbed = dynamics_simulator_perturbed.dependent_variable_history  # here you define a ditionary
# the structure of this dictionary is the folling
# keyword are the time stumps
# the values are lists of kepler elemnts at each time stamp

cartesian_perturbed_array = result2array(simulation_result_perturbed)  # convert to array
true_anomaly              = result2array(dependent_variables_perturbed)[:,-1] # take the true anomaly from the kepler elements
semimajor                 = np.mean(result2array(dependent_variables_perturbed)[:,1]) # take the semimajor axis from the kepler elements

R_jg                      = cartesian_perturbed_array[:,1:4]
distance_jg               = np.sqrt(np.sum(R_jg**2,axis=1))

ganymede_mu               = body_settings.get( 'Ganymede' ).gravity_field_settings.gravitational_parameter
jupiter_mu                = body_settings.get( 'Jupiter' ).gravity_field_settings.gravitational_parameter

ganymede_normalized_c20   = body_settings.get( 'Ganymede' ).gravity_field_settings.normalized_cosine_coefficients[2,0]
ganymede_normalized_c22   = body_settings.get( 'Ganymede' ).gravity_field_settings.normalized_cosine_coefficients[2,2]
jupiter_normalized_c20    = body_settings.get( 'Jupiter' ).gravity_field_settings.normalized_cosine_coefficients[2,0]

ganymede_reference_radius = body_settings.get( 'Ganymede' ).gravity_field_settings.reference_radius
jupiter_reference_radius  = body_settings.get( 'Jupiter' ).gravity_field_settings.reference_radius

ganymede_j2               = -ganymede_normalized_c20 * np.sqrt(5)
ganymede_unnormalized_c22 =  ganymede_normalized_c22  * np.sqrt(10)
jupiter_j2                = -jupiter_normalized_c20  * np.sqrt(5)

parameters_matrix = np.array([ganymede_j2 ,ganymede_unnormalized_c22,ganymede_reference_radius,jupiter_j2,jupiter_reference_radius,jupiter_mu])

print(ganymede_j2,ganymede_unnormalized_c22,jupiter_j2 )
## J2 from jupiter
r_dotdot1 = 3/2*jupiter_mu*jupiter_reference_radius**2/semimajor**4*jupiter_j2
dn1       = -r_dotdot1/2*np.sqrt(semimajor/jupiter_mu)
## J2 from ganymede
r_dotdot2 = 3/2*jupiter_mu*ganymede_reference_radius**2/semimajor**4*ganymede_j2
dn2       = -r_dotdot2/2*np.sqrt(semimajor/jupiter_mu)
## C22 from Ganymede
r_dotdot3 = -9*jupiter_mu*ganymede_reference_radius**2/semimajor**4*ganymede_unnormalized_c22
dn3       = -r_dotdot3/2*np.sqrt(semimajor/jupiter_mu)


print(dn1,dn2,dn3)

print()


##
directory_path = '/Users/gregorio/Desktop/DelftUni/NumericalAstro/assignments/assignment1/NumericalAstro_2021_Gregorio_Marchesini/Assignment1/OUTPUTFILES'
file1_path = '/Users/gregorio/Desktop/DelftUni/NumericalAstro/assignments/assignment1/NumericalAstro_2021_Gregorio_Marchesini/Assignment1/OUTPUTFILES/parameters.txt'
save2txt(solution=simulation_result_unperturbed,
         filename='JUICE_cartesianstate_unperturbed_Q7.dat',
         directory=directory_path
         )
save2txt(solution=dependent_variables_unperturbed,
         filename='JUICE_keplerelement_unperturbed_Q7.dat',
         directory=directory_path
         )

# w.r.t jupiter
save2txt(solution=simulation_result_perturbed,
         filename='JUICE_cartesianstate_perturbed_Q7.dat',
         directory=directory_path
         )

# w.r.t jupiter
save2txt(solution=dependent_variables_perturbed,
         filename='JUICE_keplerelement_perturbed_Q7.dat',
         directory=directory_path
         )


with open( file1_path, 'wb') as f:
        np.savetxt(file1_path ,parameters_matrix)

        
