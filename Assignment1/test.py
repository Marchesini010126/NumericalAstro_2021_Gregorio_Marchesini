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
from tudatpy.util import result2array


# # student number: 1244779 --> 1244ABC
A = 8
B = 2
C = 4

simulation_start_epoch = 33.15 * constants.JULIAN_YEAR + A * 7.0 * constants.JULIAN_DAY + B * constants.JULIAN_DAY + C * constants.JULIAN_DAY / 24.0
simulation_end_epoch   = simulation_start_epoch + 344.0 * constants.JULIAN_DAY / 24.0

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Load spice kernels.
spice.load_standard_kernels() # load the kernel?

# Create settings for celestial bodies
bodies_to_create         = ['Ganymede','Sun','Io','Callisto','Europa','Jupiter','Saturn']         # this must have a list of all the planets to create
global_frame_origin      = 'Jupiter'        # this is the origin of the refernce system
global_frame_orientation = 'ECLIPJ2000'  # orinetation of the reference system
body_settings            = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation) # body settings taken from SPICE.

# Add Ganymede exponential atmosphere
density_scale_height     = 40.0E3
density_at_zero_altitude = 2.0E-9
body_settings.get( 'Ganymede' ).atmosphere_settings = environment_setup.atmosphere.exponential( 
        density_scale_height, density_at_zero_altitude)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['Ganymede']
central_bodies      = ['Jupiter']   # body around which the propapagtion is taken


acceleration_settings_on_Ganymede_unperturbed = dict( ) # left empty on purpose

# define two distinct acceleration settings
# CASE 1 : Create global accelerations dictionary.
acceleration_settings_unperturbed = {'Ganymede':acceleration_settings_on_Ganymede_unperturbed}

# Create two distinct acceleration models.
# CASE 1
acceleration_models_unperturbed = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_unperturbed, bodies_to_propagate, central_bodies)


###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define initial state.
# identical for both the cases


initial_state_ganymede = spice.get_body_cartesian_state_at_epoch(
    target_body_name='Ganymede',
    observer_body_name='Jupiter',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time= simulation_start_epoch )

initial_state = initial_state_ganymede

# Create propagation settings for the two cases
termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )

#Case 1 : unperturbed
propagator_settings_unperturbed = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models_unperturbed,
    bodies_to_propagate,
    initial_state,
    termination_settings
)


    
# Create numerical integrator settings for both the cases
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch,
    fixed_step_size
)

###########################################################################
# PROPAGATE ORBIT #########################################################
###########################################################################

# Create simulation object and propagate dynamics

# Case 1 : unperturbed
dynamics_simulator_unperturbed = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings_unperturbed )



simulation_result_unperturbed = dynamics_simulator_unperturbed.state_history


case1_sol = result2array(simulation_result_unperturbed)
print(np.log(np.abs(case1_sol)))

print(np.shape(case1_sol))

# xg,yg,zg = case1_sol[:,7:10].reshape((3,-1))
fig,ax=plt.subplots(3,1)
xj,yj,zj = case1_sol[:,1:4].reshape((3,-1))
ax[0].plot(xj)
ax[0].set_xlabel('time')
ax[0].set_ylabel('x')
ax[1].plot(yj)
ax[0].set_xlabel('time')
ax[0].set_ylabel('y')
ax[2].plot(zj)
ax[0].set_xlabel('time')
ax[0].set_ylabel('z')
fig.suptitle('Ganymede propagation in the absence of accelerations')
plt.tight_layout()
plt.show()