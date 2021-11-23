###########################################################################
#
# # Numerical Astrodynamics 2021/2022
#
# # Assignment 1 - Propagation Settings
#
###########################################################################


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
bodies_to_create = ['Ganymede','Sun','Io','Callisto','Europa','Jupiter','Saturn']         # this must have a list of all the planets to create
global_frame_origin = 'Jupiter'        # this is the origin of the refernce system
global_frame_orientation = 'ECLIPJ2000'  # orinetation of the reference system
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation) # body settings taken from SPICE.

# Add Ganymede exponential atmosphere
density_scale_height = 40.0E3
density_at_zero_altitude = 2.0E-9
body_settings.get( 'Ganymede' ).atmosphere_settings = environment_setup.atmosphere.exponential( 
        density_scale_height, density_at_zero_altitude)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################

# Create vehicle object
bodies.create_empty_body( 'JUICE' )

# Set mass of vehicle
bodies.get_body( 'JUICE' ).mass = 2000.0
    
# Create aerodynamic coefficients interface (drag-only; zero side force and lift)
reference_area = 100.0
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area,[ drag_coefficient, 0.0 , 0.0 ] )

environment_setup.add_aerodynamic_coefficient_interface(
                bodies, 'JUICE', aero_coefficient_settings )

reference_area_radiation = reference_area
radiation_pressure_coefficient = 1.2
occulting_bodies = ["Ganymede"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)

environment_setup.add_radiation_pressure_interface(
            bodies, "JUICE", radiation_pressure_settings
)


###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies      = ['Jupiter']   



# Define accelerations acting on vehicle
# acceleration in the case of "all planets" perturbation
acceleration_settings_on_juice_all_planets = dict(
    Ganymede =[propagation_setup.acceleration.spherical_harmonic_gravity(2,2),
                propagation_setup.acceleration.aerodynamic()],  
    Jupiter  =[propagation_setup.acceleration.spherical_harmonic_gravity(4,0)],
    Saturn   =[propagation_setup.acceleration.point_mass_gravity()],
    Sun      =[propagation_setup.acceleration.point_mass_gravity(),
               propagation_setup.acceleration.cannonball_radiation_pressure()],
    Europa   =[propagation_setup.acceleration.point_mass_gravity()],
    Io       =[propagation_setup.acceleration.point_mass_gravity()],
    Callisto =[propagation_setup.acceleration.point_mass_gravity()],
)

# acceleration in the case of ganymede two body problem
acceleration_settings_on_juice_only_ganymede = dict(
    Ganymede=[propagation_setup.acceleration.point_mass_gravity()]  # create a list of possible accelerations you want. In this case you have one single body and his atmosphere
)

# Create global accelerations dictionary for the two cases
acceleration_settings_all_planets   = {'JUICE': acceleration_settings_on_juice_all_planets}
acceleration_settings_only_ganymede = {'JUICE': acceleration_settings_on_juice_only_ganymede}

# Create acceleration models.
acceleration_models_all_planets = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_all_planets, bodies_to_propagate, central_bodies)
acceleration_models_only_ganymede = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_only_ganymede, bodies_to_propagate, central_bodies)

###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define initial state.
system_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name       ='JUICE',
    observer_body_name     ='Jupiter',
    reference_frame_name   ='ECLIPJ2000',
    aberration_corrections ='NONE',
    ephemeris_time= simulation_start_epoch )

# Define required outputs (for both cases is the same)
dependent_variables_to_save = [
    propagation_setup.dependent_variable.relative_position('Ganymede','Jupiter')
]


# Create propagation settings.
termination_settings            = propagation_setup.propagator.time_termination( simulation_end_epoch )
propagator_settings_all_planets = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models_all_planets,
    bodies_to_propagate,
    system_initial_state,
    termination_settings,
    output_variables = dependent_variables_to_save
)
propagator_settings_only_ganymede = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models_only_ganymede,
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
dynamics_simulator_all_planets = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings_all_planets )

# Create simulation object and propagate dynamics.
dynamics_simulator_only_ganymede = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings_only_ganymede)

simulation_result_all_planets       = dynamics_simulator_all_planets.state_history
dependent_variables_all_planets     = dynamics_simulator_all_planets.dependent_variable_history  # here you define a ditionary
simulation_result_only_ganymede     = dynamics_simulator_only_ganymede.state_history
dependent_variables_only_ganymede   = dynamics_simulator_only_ganymede.dependent_variable_history  # here you define a ditionary
# the structure of this dictionary is the following
# keyword are the time stumps
# the values are lists of kepler elemnts at each time stamp

save2txt(solution=dependent_variables_all_planets,
         filename='Ganymede_wrt_Jupiter_perturbed_case_Q5iv.dat',
         directory='./OUTPUTFILES'
         )

# w.r.t jupiter
save2txt(solution=simulation_result_all_planets,
         filename='JUICE_cartesianstate_Q5iv.dat',
         directory='./OUTPUTFILES'
         )


save2txt(solution=dependent_variables_only_ganymede,
         filename='Ganymede_wrt_Jupiter_unperturbed_case_Q5ii.dat',
         directory='./OUTPUTFILES'
         )

# w.r.t jupiter
save2txt(solution=simulation_result_only_ganymede,
         filename='JUICE_cartesianstate_Q5ii.dat',
         directory='./OUTPUTFILES'
         )
