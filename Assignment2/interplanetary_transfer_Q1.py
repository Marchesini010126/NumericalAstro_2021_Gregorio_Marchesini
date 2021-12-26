
''' 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

from plotly import express as px
import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
from interplanetary_transfer_helper_functions import * 

# imports all the name space in this way
# this means that also the variables name of the module
# are loaded. This is risky in normal cases but it works in this 
# case

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"
TXTfile_to_save = 'CartesianResults_AE4868_2021_2_5622824.dat'
###########################################################################
# RUN CODE FOR QUESTION 1 #################################################
###########################################################################

# list of natural bodies to create 
frame_origin         = 'Sun'
bodies_to_create     = ['Sun','Mars','Earth','Venus']
vehicle_specific     = {'name':'Spacecraft','mass':2000}   # name / mass kg

# Create body objects
bodies = create_simulation_bodies( bodies_to_create,frame_origin,vehicle_spec=vehicle_specific)

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

# Create propagation settings and propagate dynamics
dynamics_simulator = propagate_trajectory( departure_epoch, arrival_epoch, bodies, lambert_arc_ephemeris,
                     use_perturbations = False)
write_propagation_results_to_file(
    dynamics_simulator, lambert_arc_ephemeris, "Q1",output_directory)

# Extract state history from dynamics simulator
state_history = dynamics_simulator.state_history

# Evaluate the Lambert arc model at each of the epochs in the state_history
lambert_history = get_lambert_arc_history( lambert_arc_ephemeris, state_history )


def history2array(state_history):
    state = np.array(list(state_history.values()))
    time  = np.array(list(state_history.keys())).reshape((-1,1))
    res = np.hstack((time,state))
    return res

def history2DataFrame(state_history,classes):
    sol= history2array(state_history)
    frame = pd.DataFrame(sol,columns=classes)
    return frame

############################################################
## plotting phase
############################################################

columns                      = ['time (s)','x [m]','y [m]','z [m]','V_x [m/s]','V_y [m/s]','V_z [m/s]']
lambert_cartesian_state_df   = history2DataFrame(lambert_history,columns)
numerical_cartesian_state_df = history2DataFrame(state_history,columns)
lambert_cartesian_state      = history2array(lambert_history)
numericak_cartesian_state    = history2array(state_history)
time                         = (lambert_cartesian_state[:,0]-lambert_cartesian_state[0,0])/60/60/24




## initialise SUN plot
resolution = 100 
radius  =5.7E9  
theta   = np.linspace(0,2*np.pi,resolution)
psi     = np.linspace(0,np.pi,resolution)
x       = radius*np.outer(np.cos(theta),np.sin(psi))
y       = radius*np.outer(np.sin(theta),np.sin(psi))
z       = np.tile(radius*np.cos(psi),(resolution,1))



# Initialise figure 
# exrecise 1A : 3d orbit
figure1 = px.line_3d(lambert_cartesian_state_df,x= 'x [m]',y = 'y [m]', z = 'z [m]')
figure1.add_trace(go.Surface(x=x, y=y, z=z,showscale=False))
figure1.add_trace(go.Scatter3d(
    x =[lambert_cartesian_state[0,1],lambert_cartesian_state[-1,1]],
    y =[lambert_cartesian_state[0,2],lambert_cartesian_state[-1,2]],
    z = [lambert_cartesian_state[0,3],lambert_cartesian_state[-1,3]],
    mode='text+markers',
    text = ['Earth','Venus'],
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )))

figure1.update_layout(
    scene_camera = dict(
    eye=dict(x=1.6, y=1.6, z=1.9)),
    scene = dict(
        xaxis = dict(nticks=4,showexponent = 'all',
        exponentformat = 'e',range=[1.5E11, -1.5E11],showline=True, linewidth=2, linecolor='black'),
        
        yaxis = dict(nticks=4,showexponent = 'all',
        exponentformat = 'e',range=[1.5E11, -1.5E11],showline=True, linewidth=2, linecolor='black'),
        
        zaxis = dict(nticks=4,showexponent = 'all',
        exponentformat = 'e',range=[1.5E11, -1.5E11],showline=True, linewidth=2, linecolor='black'),
        aspectmode='cube', 
        aspectratio=dict(x=1, y=1, z=0.95)),
    
        width=700,
        margin=dict(r=20, l=10, b=10, t=10),
        showlegend=False)


figure1.show()
figure1.write_image("output_images/exercise1_full_orbit.eps")

# plot difference between the two orbits 

col                 = ['time','x','y','z','vx','vy','vz']
solution_difference = lambert_cartesian_state-numericak_cartesian_state
difference_df       = pd.DataFrame(solution_difference,columns=col)

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=solution_difference[:,1],
                    mode='lines+markers',
                    name=r'$\Delta x$'))
fig.add_trace(go.Scatter(x=time, y=solution_difference[:,2],
                    mode='lines+markers',
                    name=r'$\Delta y$'))
fig.add_trace(go.Scatter(x=time, y=solution_difference[:,3],
                    mode='lines+markers', name=r'$\Delta z$'))

fig.update_layout(
    xaxis_title="time [days]",
    yaxis_title=r"$\Delta [m]$",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="RebeccaPurple"),
    width = 1000)
fig.show()

fig.write_image("output_images/exercise1_single_component_error.eps")

for 