clear all
close all

filename = 'cartesian_results_AE4868_2021_A1_5622824.txt';
system(['rm ./',filename])

addpath('/Users/gregorio/Desktop/DelftUni/NumericalAstro/assignments/assignment1/NumericalAstro_2021_Gregorio_Marchesini/Assignment1/OUTPUTFILES')
% question 1
AnalityicalOmega        = dlmread('OUTPUTFILES/OmegaAnalyticalDynamics.txt');
AnalityicalomegaSmall   = dlmread('OUTPUTFILES/omegaSmallAnalyticalDynamics.txt');
CartesianCoordinatesQ1  = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q1.dat');
KeplerianStateQ1        = dlmread('OUTPUTFILES/JUICE_KeplerElements_Q1.dat');

raw1 = CartesianCoordinatesQ1(end,:);
% question 2
CartesianCoordinatesQ2  = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q2.dat');
KeplerianStateQ2        = dlmread('OUTPUTFILES/JUICE_KeplerElemets_Q2.dat');
raw2 = CartesianCoordinatesQ2(end,:);
% question 3

CartesianCoordinatesQ3  = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q3.dat');
accelration_types       = dlmread('OUTPUTFILES/JUICE_accelerations_Q3.dat');
raw3 = CartesianCoordinatesQ3(end,:);
%question 4

% unperturbed_orbit_case
cartesian_unperturbed = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q2.dat');
% case i
cartesian_casei       = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q4I.dat');
raw4 = cartesian_casei(end,:);
% case ii
cartesian_caseii      = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q4II.dat');
raw5 = cartesian_caseii(end,:);

%question 5


Juice_unperturbed_Jframe    = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q5ii.dat');
Juice_perturbed_Jframe      = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q5iv.dat');
Ganymede_unperturbed_Jframe = dlmread('OUTPUTFILES/Ganymede_wrt_Jupiter_unperturbed_case_Q5ii.dat');
Ganymede_perturbed_Jframe   = dlmread('OUTPUTFILES/Ganymede_wrt_Jupiter_perturbed_case_Q5iv.dat');

Juice_unperturbed_Gframe_Jpropagation  = Juice_unperturbed_Jframe(:,2:end) - Ganymede_unperturbed_Jframe(:,2:end) ; % juice position in Ganymede refernce, but obtained from the propagation around jupiter
Juice_perturbed_Gframe_Jpropagation    = Juice_perturbed_Jframe(:,2:end) -  Ganymede_perturbed_Jframe(:,2:end);

Juice_unperturbed_Gframe_Jpropagation = [Juice_unperturbed_Jframe(:,1),Juice_unperturbed_Gframe_Jpropagation];
Juice_perturbed_Gframe_Jpropagation   = [Juice_perturbed_Jframe(:,1),Juice_perturbed_Gframe_Jpropagation];

raw6 = Juice_unperturbed_Gframe_Jpropagation(end,:);
raw7 = Juice_perturbed_Gframe_Jpropagation(end,:);
%%
%question 6

Juice_and_Ganymede_unperturbed_Jframe    = dlmread('OUTPUTFILES/JUICE_cartesianstate_unperturbed_Q6.dat');
Juice_and_Ganymede_perturbed_Jframe      = dlmread('OUTPUTFILES/JUICE_cartesianstate_perturbed_Q6.dat');

% cartesian coordinates
Juice_unperturbed_Jframe                   = Juice_and_Ganymede_unperturbed_Jframe(:,2:7);
Juice_perturbed_Jframe                     = Juice_and_Ganymede_perturbed_Jframe(:,2:7);

Ganymede_unperturbed_Jframe_Jpropagation   = Juice_and_Ganymede_unperturbed_Jframe(:,8:end);
Juice_unperturbed_Gframe_Jpropagation      = Juice_unperturbed_Jframe - Ganymede_unperturbed_Jframe_Jpropagation; % juice position in Ganymede refernce, but obtained from the propagation around jupiter
Ganymede_perturbed_Jframe_Jpropagation     = Juice_and_Ganymede_perturbed_Jframe(:,8:end);
Juice_perturbed_Gframe_Jpropagation        = Juice_perturbed_Jframe - Ganymede_perturbed_Jframe_Jpropagation;


Juice_unperturbed_Jframe              = [Juice_and_Ganymede_unperturbed_Jframe(:,1),Juice_unperturbed_Jframe];
Juice_unperturbed_Gframe_Jpropagation = [Juice_and_Ganymede_unperturbed_Jframe(:,1),Juice_unperturbed_Gframe_Jpropagation];
Juice_perturbed_Jframe                = [Juice_and_Ganymede_unperturbed_Jframe(:,1),Juice_perturbed_Jframe];
Juice_perturbed_Gframe_Jpropagation   = [Juice_and_Ganymede_unperturbed_Jframe(:,1),Juice_perturbed_Gframe_Jpropagation];

raw8  = Juice_unperturbed_Gframe_Jpropagation(end,:);
raw9  = Juice_unperturbed_Jframe(end,:);
raw10 = Juice_perturbed_Gframe_Jpropagation(end,:);
raw11 = Juice_perturbed_Jframe(end,:);


savematrix = [raw1;
              raw2;
              raw3;
              raw4;
              raw5;
              raw6;
              raw7;
              raw8;
              raw9;
              raw10;
              raw11];
%%
format long
writematrix(double(savematrix),filename,'Delimiter','space')


