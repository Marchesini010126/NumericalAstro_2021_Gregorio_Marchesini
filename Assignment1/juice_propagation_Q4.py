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
bodies_to_create = ['Ganymede','Jupiter']         # this must have a list of all the planets to create
global_frame_origin = 'Ganymede'        # this is the origin of the refernce system
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


###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies      = ['Ganymede']   # bodies which are contributing with the full gravity and are not only perturbation



# Define accelerations acting on vehicle.
acceleration_settings_on_juice_questionI = dict(
    Ganymede =[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter  =[propagation_setup.acceleration.spherical_harmonic_gravity(4,0)],
)

acceleration_settings_on_juice_questionII = dict(
    Ganymede =[propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.aerodynamic()],  
)


# do note that when calling a method from a class you need to put the parenthesis. 
# Otherwise this will rise an error since the method is defined and not the call to the function output

ganymede_mu               = body_settings.get( 'Ganymede' ).gravity_field_settings.gravitational_parameter
ganymede_normalized_c20   = body_settings.get( 'Ganymede' ).gravity_field_settings.normalized_cosine_coefficients[2,0]
ganymede_reference_radius = body_settings.get( 'Ganymede' ).gravity_field_settings.reference_radius
ganymede_j2               = -ganymede_normalized_c20 * np.sqrt(5)

# Create global accelerations dictionary.
acceleration_settingsI  = {'JUICE': acceleration_settings_on_juice_questionI}
acceleration_settingsII = {'JUICE': acceleration_settings_on_juice_questionII}

# Create acceleration models.
acceleration_modelsI = propagation_setup.create_acceleration_models(
        bodies, acceleration_settingsI, bodies_to_propagate, central_bodies)

acceleration_modelsII = propagation_setup.create_acceleration_models(
        bodies, acceleration_settingsII, bodies_to_propagate, central_bodies)

###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define initial state.
system_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='JUICE',
    observer_body_name='Ganymede',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time= simulation_start_epoch )

## harmonics LIST

sph_list = [(b,0) for b in range(0,5)]; # juppiter m_max = 0 l_max = 12

# Define required outputs

dependent_variables_to_saveI = [
    propagation_setup.dependent_variable.single_acceleration_norm(
    propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Ganymede'),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration(
    'JUICE','Jupiter',sph_list),
]

dependent_variables_to_saveII = [
    propagation_setup.dependent_variable.single_acceleration_norm(
    propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Ganymede'),
    propagation_setup.dependent_variable.single_acceleration(
    propagation_setup.acceleration.aerodynamic_type,'JUICE','Ganymede') 
]

#[aerodynamic(Ganymede),point_mass_gravity_type(Europa),point_mass_gravity_type(Callisto),point_mass_gravity_type(IO),point_mass_gravity_type(SUN),Solar_radiation(Sun),Spherical_harmonics(ganymede((00),(20),(22))),Spherical_harmonics(Jupiter((00),(20),(40)))]

# note here you made the error of not creating a list. you should create  alist of outputs.
# Read carefully the errors since they are really explicits when you know the type of each variable.


# Create propagation settings.
termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )

propagator_settingsI = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_modelsI,
    bodies_to_propagate,
    system_initial_state,
    termination_settings,
    output_variables = dependent_variables_to_saveI
)

propagator_settingsII = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_modelsII,
    bodies_to_propagate,
    system_initial_state,
    termination_settings,
    output_variables = dependent_variables_to_saveII
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
dynamics_simulatorI = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settingsI )

# Create simulation object and propagate dynamics.
dynamics_simulatorII = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settingsII )

simulation_resultI = dynamics_simulatorI.state_history
simulation_resultII = dynamics_simulatorII.state_history

dependent_variablesI  = dynamics_simulatorI.dependent_variable_history  # here you define a ditionary
dependent_variablesII = dynamics_simulatorII.dependent_variable_history  # here you define a ditionary
# the structure of this dictionary is the folling
# keyword are the time stumps
# the values are lists of kepler elemnts at each time stamp

###########################################################################
# SAVE RESULTS ############################################################
###########################################################################

save2txt(solution=simulation_resultI,
         filename='JUICEPropagationHistory_Q4I.dat',
         directory='./'
         )

save2txt(solution=dependent_variablesI,
         filename='JUICEPropagationHistory_DependentVariables_Q4I.dat',
         directory='./'
         )
save2txt(solution=simulation_resultII,
         filename='JUICEPropagationHistory_Q4II.dat',
         directory='./'
         )

save2txt(solution=dependent_variablesII,
         filename='JUICEPropagationHistory_DependentVariables_Q4II.dat',
         directory='./'
         )

save2txt(solution=simulation_resultI,
         filename='JUICE_cartesianstate_Q4I.dat',
         directory='./OUTPUTFILES'
         )

save2txt(solution=dependent_variablesI,
         filename='JUICE_accelerations_Q4I.dat',
         directory='./OUTPUTFILES'
         )

save2txt(solution=simulation_resultII,
         filename='JUICE_cartesianstate_Q4II.dat',
         directory='./OUTPUTFILES'
         )

save2txt(solution=dependent_variablesII,
         filename='JUICE_accelerations_Q4II.dat',
         directory='./OUTPUTFILES'
         )


###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# import matplotlib.ticker as mticker

# # Extract time and Kepler elements from dependent variables
# kepler_elements = np.vstack(list(dependent_variables.values()))
# time            = np.array(list(dependent_variables.keys()))
# time_days       = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

# ## only if you want to plot in python

# # fig,ax     =plt.subplots(3,2,figsize=(10,6))
# # elements   =[r'$a\left[km\right]    $',r'$e$',r'$i\left[rad\right]$',r'$\omega\left[rad\right]$',r'$\Omega\left[rad\right]$',r'$\theta\left[rad\right]$']
# # indxmatrix =np.arange(6).reshape(3,2)
# # f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
# # g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
# # for ii in range(3):
# #     for jj in range(2):
# #       ax[ii,jj].plot(time_days,kepler_elements[:,indxmatrix[ii,jj]])
# #       ax[ii,jj].set_xlabel('t [days]')
# #       ax[ii,jj].set_xlabel(elements[indxmatrix[ii][jj]])
# #       ax[ii,jj].get_yaxis().get_major_formatter().set_useOffset(False)
# #       ax[ii,jj].set_yticks(np.linspace(min(kepler_elements[:,indxmatrix[ii,jj]]),max(kepler_elements[:,indxmatrix[ii,jj]]),4))
# # plt.tight_layout()
# # plt.show()

