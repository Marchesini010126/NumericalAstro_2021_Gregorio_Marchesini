 %% This script is used to plot all the simulation results obtained from the 
%% RUN Python CODE AND DOWLOAD ALL THE FILES
close all
clear all
%%
% Run all python scripts ?

RunPythonScriptNumber = 0; % ['all',0,1,2,3,4,5,6,7]
visible = 'off';           % off images to show
print_images = 1;          % [0 - 1] [no yes]
% all --> run all
%  0  --> none
% 1) script 1
% 2) script 2
% 3) script 3
% 4) script 4
% 5) script 5
% 6) script 6
% 7) script 7

% INSERT HERE THE PATH OF OUTPUT
% FILES WHERE YOU STORE ALL THE 
% OUTPUTS FROM THE PYTHON CODE

addpath('/Users/gregorio/Desktop/DelftUni/NumericalAstro/assignments/assignment1/NumericalAstro_2021_Gregorio_Marchesini/Assignment1/OUTPUTFILES')
addpath('./matlabUtilities')

% Specify python environment that you want to use
% virtual enviroonment for your TuDat package
pythonInterprester  = '/Users/gregorio/Desktop/DelftUni/NumericalAstro/venv/bin/python';
% folder for your python scripts
pythonScriptsFolder = '/Users/gregorio/Desktop/DelftUni/NumericalAstro/assignments/assignment1/NumericalAstro_2021_Gregorio_Marchesini/Assignment1';

% python scripts name
pythonScript1      = fullfile(pythonScriptsFolder,'juice_propagation_Q1.py');
pythonScript2      = fullfile(pythonScriptsFolder,'juice_propagation_Q2.py');
pythonScript3      = fullfile(pythonScriptsFolder,'juice_propagation_Q3.py');
pythonScript4      = fullfile(pythonScriptsFolder,'juice_propagation_Q4.py');
pythonScript5      = fullfile(pythonScriptsFolder,'juice_propagation_Q5.py');
pythonScript6      = fullfile(pythonScriptsFolder,'juice_propagation_Q6.py');
pythonScript7      = fullfile(pythonScriptsFolder,'juice_propagation_Q7.py');

% listing
pythonScripts = {pythonScript1,pythonScript2,pythonScript3,pythonScript4,pythonScript5,pythonScript6,pythonScript7};

% define figures folder and saving parameters
imagesFolder = './output_images';

% run python scripts

if strcmp(RunPythonScriptNumber,'all')
    % run the python scripts
    for jj=1:length(pythonScripts)
        fprintf('Running Script number : %i\n',jj)
        system([pythonInterprester,' ',pythonScripts{jj}]);
    end
elseif RunPythonScriptNumber~= 0
    fprintf('Running Script number : %i\n',RunPythonScriptNumber)
    system([pythonInterprester,' ',pythonScripts{RunPythonScriptNumber}]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% QUESTION 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% section A

AnalityicalOmega        = dlmread('OUTPUTFILES/OmegaAnalyticalDynamics.txt');
AnalityicalomegaSmall   = dlmread('OUTPUTFILES/omegaSmallAnalyticalDynamics.txt');
CartesianCoordinatesQ1  = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q1.dat');
KeplerianStateQ1        = dlmread('OUTPUTFILES/JUICE_KeplerElements_Q1.dat');

time_days               = (KeplerianStateQ1(:,1)-KeplerianStateQ1(1,1))/60/60/24; %days
kep_labels = {'a [m]','e','i [rad]','\omega [rad]','\Omega [rad]','\theta[rad]'};

fig_task1a=figure('Position',[50,50,900,600],'visible',visible);
for jj=1:6
    
    ax = myaxes(subplot(3,2,jj));
    hold on
    plot(ax,time_days,KeplerianStateQ1(:,jj+1));
    
    if strcmp(kep_labels{jj},'\Omega [rad]')
        plot(ax,time_days,AnalityicalOmega,'DisplayName','Analytical J2 effect Trend');
    end
    if strcmp(kep_labels{jj},'\omega [rad]')
        plot(ax,time_days,AnalityicalomegaSmall,'DisplayName','Analytical J2 effect Trend');
    end
    
    xlabel('days');
    ylabel(kep_labels{jj});
end


% Notes by greg
% the trand for the RAAN is not following the real secular trend(we need to discover why this is the case)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% QUESTION 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% section A

CartesianCoordinatesQ2  = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q2.dat');
KeplerianStateQ2_full   = dlmread('OUTPUTFILES/JUICE_KeplerElemets_Q2.dat');
KeplerianStateQ2        = KeplerianStateQ2_full(:,1:7);

acceleration_q2         = KeplerianStateQ2_full(:,8:end); % m/s^2 acc norm
time_sec   =  KeplerianStateQ2(:,1)-KeplerianStateQ2(1,1);          %s
time_days  = (KeplerianStateQ2(:,1)-KeplerianStateQ2(1,1))/60/60/24; %days
kep_labels = {'\Delta a [m]','\Delta e','\Delta i [rad]','\Delta \omega [rad]','\Delta \Omega [rad]','\Delta \theta[rad]'};

fig_task2a  = figure('Position',[50,50,900,600],'Name','Task B Stability of the integrator for Point Mass planet','visible',visible);

mass_ganymede = 1.4819e23;                      % kg
G             = 6.67e-20;                       % km3/kg/s2
mu_ganymede   = G*mass_ganymede;                % km3/s2
semimajor0    = KeplerianStateQ2(1,2)*10^-3;    % km
mean_motion   = sqrt(mu_ganymede/semimajor0^3); % 1/s
period        = 1/mean_motion*2*pi;
theta_time    = KeplerianStateQ2(1,7) + mean_motion*time_sec;

for jj=1:6
    ax = myaxes(subplot(3,2,jj));
    hold on
    
    plot(ax,time_days,abs(KeplerianStateQ2(1,jj+1)-KeplerianStateQ2(:,jj+1)));  
    xlabel('days');
    ylabel(kep_labels{jj});
end


%%
%%%%%%%%%%%%%%%%% plot ganymede
Ganymede_figure = figure('Name','Question 2A','visible',visible);
hold on
[x,y,z] = sphere;
image   = imread('ganymedeimage.jpeg');
planets = surf(x*2634100,...
               y*2634100,...
               z*2634100,...
               flipud(image),...
    'FaceColor','texturemap',...
    'EdgeColor','none');



plot3(CartesianCoordinatesQ2(:,2),CartesianCoordinatesQ2(:,3),CartesianCoordinatesQ2(:,4),'linewidth',2);
axis equal
ax=gca();
view(14,29)
%%%%%%%%%%%%%%%%%% End plot ganymede

% if the numerical integrator is stable the error in comparison to the
% initial state should be zero. But this is not going to be the case 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CartesianCoordinatesQ3  = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q3.dat');
accelration_types       = dlmread('OUTPUTFILES/JUICE_accelerations_Q3.dat');
% Accelration list 
% first column is always the time 

%1 time
% class i

% 2 aerodynamic(Ganymede)
% 3 point_mass_gravity_type(Europa)
% 4 point_mass_gravity_type(Callisto)
% 5 point_mass_gravity_type(Saturn)
% 6 point_mass_gravity_type(IO)
% 7 point_mass_gravity_type(SUN)
% 8 Solar_radiation(Sun)

%class_ii
% 9  Spherical_harmonics(ganymede((00))
% 10 Spherical_harmonics(ganymede((20)),
% 11 Spherical_harmonics(ganymede((22)),
% 12 Spherical_harmonics(Jupiter((00))
% 13 Spherical_harmonics(Jupiter((20),
% 14 Spherical_harmonics(Jupiter((40),

class_i={'aerodynamic(Ganymede)','point_mass_gravity_type(Europa)',...
        'point_mass_gravity_type(Callisto)','point_mass_gravity_type(Saturn)',...
        'point_mass_gravity_type(Io)','point_mass_gravity_type(Sun)',...
        'Solar_radiation(Sun)'};
    
class_ii={'Spherical_harmonics(Ganymede,(00))',...
        'Spherical_harmonics(Ganymede,(20))','Spherical_harmonics(Ganymede,(22))',...
        'Spherical_harmonics(Jupiter,(00))','Spherical_harmonics(Jupiter,(20))',...
        'Spherical_harmonics(Jupiter,(40))'};

time_sec  = CartesianCoordinatesQ3(:,1)-CartesianCoordinatesQ3(1,1);
time_days = (CartesianCoordinatesQ3(:,1)-CartesianCoordinatesQ3(1,1))/60/60/24;

fig_task3ai_ii = figure('Name','Question 3ai','Position',[0,0,1300,700],'visible',visible);

ax1  = subplot(2,1,1);
semilogy(ax1,time_days,accelration_types(:,2:8),'linewidth',2)
ax1  = myaxes(ax1);

hold on
xlabel('days')
ylabel('Acceleration (m/s^2)')

legend(class_i,'Interpreter','none','location','bestoutside')

ax2  = subplot(2,1,2);
semilogy(ax2,time_days,accelration_types(:,9:14),'linewidth',2)
ax2  = myaxes(ax2);

ylim([1e-12,1e2])
hold on
xlabel('days')
ylabel('Acceleration (m/s^2)')
legend(class_ii,'Interpreter','none','location','bestoutside')
%%
fig_task3kepler = figure('Name','Question 3 : kepler elements','Position',[0,0,1300,700],'visible','on');

sun_distance = sqrt(sum((accelration_types(:,15:17)-accelration_types(:,18:20)).^2,2));
sun_distance = sun_distance/max(sun_distance);

vertical=101.503:1.53:115;

ax = myaxes(subplot(411));
hold on
plot(ax,time_days*24,accelration_types(:,21),'LineWidth',2);
arrayfun(@(x)xline(ax,x,'--k','linewidth',1,'HandleVisibility','off'),vertical)
xlabel('hours');
ylabel('semimajor axis [m]');
xlim([100,115])

ax = myaxes(subplot(412));
hold on
plot(ax,time_days*24,accelration_types(:,25),'LineWidth',2);
arrayfun(@(x)xline(ax,x,'--k','linewidth',1,'HandleVisibility','off'),vertical)
xlabel('hours');
ylabel('\Omega [rad]');
xlim([100,115])

ax = myaxes(subplot(413));
hold on
plot(ax,time_days*24,sun_distance,'LineWidth',2);
arrayfun(@(x)xline(ax,x,'--k','linewidth',1,'HandleVisibility','off'),vertical(1:2:end))
xlabel('hours');
ylabel('||r_{Ss}-r_{SG}||_N');
xlim([100,115])

vertical=101.772:1.53:115;
ax = myaxes(subplot(414));
hold on
plot(ax,time_days*24,accelration_types(:,7)/max(accelration_types(:,7)),'LineWidth',2);
arrayfun(@(x)xline(ax,x,'--k','linewidth',1,'HandleVisibility','off'),vertical)
xlabel('hours');
ylabel('a_{SG} \\ max(a_{SG})');
xlim([100,115]) 

annotation(fig_task3kepler,'doublearrow',[0.209230769230769 0.286153846153846],...
    [0.886142857142857 0.885714285714286]);

% Create doublearrow
annotation(fig_task3kepler,'doublearrow',[0.209230769230769 0.286153846153846],...
    [0.659 0.658571428571429]);

% Create doublearrow
annotation(fig_task3kepler,'doublearrow',[0.209230769230769 0.363076923076923],...
    [0.449 0.448571428571429]);

% Create doublearrow
annotation(fig_task3kepler,'doublearrow',[0.223846153846154 0.299230769230769],...
    [0.224285714285714 0.224285714285714]);

% Create textbox
annotation(fig_task3kepler,'textbox',...
    [0.226384615384615 0.89 0.0597692307692307 0.0285714285714287],...
    'String',{'1.53 hours'},...
    'FitBoxToText','off');

% Create textbox
annotation(fig_task3kepler,'textbox',...
    [0.217923076923077 0.67 0.0597692307692307 0.0285714285714287],...
    'String',{'1.53 hours'},...
    'FitBoxToText','off');

% Create textbox
annotation(fig_task3kepler,'textbox',...
    [0.236384615384615 0.158571428571429 0.0597692307692308 0.0285714285714287],...
    'String',{'1.53 hours'},...
    'FitBoxToText','off');

% Create textbox
annotation(fig_task3kepler,'textbox',...
    [0.258692307692308 0.388571428571429 0.0597692307692307 0.0285714285714287],...
    'String','3.06 hours',...
    'FitBoxToText','off');



%%
% task 3d

fig_task3d = figure('Name','Question 3d','Position',[0,0,1300,300],'visible',visible);
semilogy(time_days,accelration_types(:,6:7),'linewidth',2)
ax = myaxes(gca());
hold on
ylim([1e-10,1e-6])
xticks([0:max(time_days)])
xlabel('days')
ylabel('Accelration (m/s^2)')
legend('Io','Sun','location','best')


vertical=0.3:2.35:15;
arrayfun(@(x)xline(ax,x,'--k','linewidth',1,'HandleVisibility','off'),vertical)

% Create doublearrow
annotation(fig_task3d,'doublearrow',[0.149230769230769 0.263846153846154],...
    [0.802333333333333 0.8]);

% Create textbox
annotation(fig_task3d,'textbox',...
    [0.180230769230769 0.816666666666667 0.0597692307692308 0.056666666666667],...
    'String',{'2.3 days'},...
    'FitBoxToText','off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% unperturbed_orbit_case
cartesian_unperturbed = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q2.dat');
% case i
cartesian_casei       = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q4I.dat');
acc_casei             = dlmread('OUTPUTFILES/JUICE_accelerations_Q4I.dat');
% case ii
cartesian_caseii      = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q4II.dat');
acc_caseii            = dlmread('OUTPUTFILES/JUICE_accelerations_Q4II.dat');

delta_r_casei  = cartesian_unperturbed-cartesian_casei;
delta_r_caseii = cartesian_unperturbed-cartesian_caseii;

delta_r_casei  = sqrt(sum(delta_r_casei(:,2:4).^2,2));
delta_r_caseii = sqrt(sum(delta_r_caseii(:,2:4).^2,2));

fig_task4a=figure('Name','Question 4a','Position',[50,50,1000,400],'visible',visible);

semilogy(time_days,delta_r_casei,'LineWidth',2,'DisplayName','case i : Jupiter spherical harmonics')
hold on
semilogy(time_days,delta_r_caseii,'LineWidth',2,'DisplayName','case ii : Ganymede atmospheric effect')
ax = myaxes(gca());

grid on
xticks(0:max(time_days))
xlabel('days')
ylabel('displacement norm (m)')
legend('location','best')


%%
% estimate integrals for all the accelrations
% note all the components are present 


%case i
%[Point_mass_gravity(Ganymede)_norm,Point_mass_gravity(Jupyter),Spherical_Harmonics(Jupiter,(20),Spherical_Harmonics(Jupiter,(40)))]

sph_acc_10_casei = acc_casei(:,3:5);
sph_acc_20_casei = acc_casei(:,6:8);
sph_acc_30_casei = acc_casei(:,9:11);

sph_acc_total_casei = sph_acc_10_casei+sph_acc_20_casei+sph_acc_30_casei;

perturbation_acc_integral_casei   = cumtrapz(time_days,sqrt(sum(sph_acc_total_casei.^2,2)));
efficiency_casei     = delta_r_casei./perturbation_acc_integral_casei;

% indentify speed and radial direction
% define a refernce frame that is always aligned with the speed and the 
% radial position of the space_craft 

speed_direction_casei            = cartesian_casei(:,5:end)./ sqrt(sum(cartesian_casei(:,5:end).^2,2));
radial_direction_casei           = cartesian_casei(:,2:4)./ sqrt(sum(cartesian_casei(:,2:4).^2,2));
angular_momentum_direction_casei = cross(radial_direction_casei,speed_direction_casei);
angular_momentum_direction_casei = angular_momentum_direction_casei./sqrt(sum(angular_momentum_direction_casei.^2,2));
along_track_direction_casei      = cross(angular_momentum_direction_casei,radial_direction_casei);

along_track_component_casei  = sum(sph_acc_total_casei.*along_track_direction_casei,2);
outofplane_component_casei   = sum(sph_acc_total_casei.*angular_momentum_direction_casei,2);
radial_component_casei             = sum(sph_acc_total_casei.*radial_direction_casei,2);


% case ii
%[Point_mass_gravity(Ganymede),Aerodynamic(Ganymede)]

aerodynamic_caseii =  acc_caseii(:,3:end);
perturbation_acc_integral_caseii  = cumtrapz(time_days,sqrt(sum(aerodynamic_caseii.^2,2)));

efficiency_caseii    = delta_r_caseii./perturbation_acc_integral_caseii;

% indentify speed and radial direction
% define a refernce frame that is always aligned with the speed and the 
% radial position of the space_craft 


speed_direction_caseii            = cartesian_caseii(:,5:end)./ sqrt(sum(cartesian_caseii(:,5:end).^2,2));
radial_direction_caseii           = cartesian_caseii(:,2:4)./ sqrt(sum(cartesian_caseii(:,2:4).^2,2));
angular_momentum_direction_caseii = cross(radial_direction_caseii,speed_direction_caseii);
angular_momentum_direction_caseii = angular_momentum_direction_caseii./sqrt(sum(angular_momentum_direction_caseii.^2,2));
along_track_direction_caseii      = cross(angular_momentum_direction_caseii,radial_direction_caseii);

along_track_component_caseii     = sum(aerodynamic_caseii.*along_track_direction_caseii,2);
outofplane_component_caseii      = sum(aerodynamic_caseii.*angular_momentum_direction_caseii,2);
radial_component_caseii          = sum(aerodynamic_caseii.*radial_direction_caseii,2);


fig_task4b_effciency=figure('Name','Question 4b efficinecy','Position',[50,50,1000,400],'visible',visible);

semilogy(time_days,efficiency_casei,'linewidth',2,'DisplayName','Jupiter spherical harmonics');
hold on
semilogy(time_days,efficiency_caseii,'linewidth',2,'DisplayName','Ganymede Atmospheric Drag');
ax=myaxes(gca());
grid on
xlabel('days')
ylabel('efficency (s)')
legend('location','best')


%%

acc_matrix = [radial_component_casei,along_track_component_casei,outofplane_component_casei,radial_component_caseii,along_track_component_caseii,outofplane_component_caseii];
titles     = {'radial component SPH','along-track component SPH','out-of-plane component SPH','radial-Drag','along-track Drag','out-of-plane Drag'};
fig_task4b_com=figure('Name','Question 4b components','Position',[50,50,1500,700],'visible',visible);


for ii = 1:6
    subplot(2,3,ii)
    plot(time_days,acc_matrix(:,ii),...
        'r','linewidth',2);
    ax=myaxes(gca());
    hold on
    grid on
    xlabel('time (days)')
    ylabel('acceleartion (m/s^2)')
    title(titles(ii))
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Question 5
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% cartesian coordinates
Juice_unperturbed_Jframe    = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q5ii.dat');
Juice_perturbed_Jframe      = dlmread('OUTPUTFILES/JUICE_cartesianstate_Q5iv.dat');
Ganymede_unperturbed_Jframe = dlmread('OUTPUTFILES/Ganymede_wrt_Jupiter_unperturbed_case_Q5ii.dat');
Ganymede_perturbed_Jframe   = dlmread('OUTPUTFILES/Ganymede_wrt_Jupiter_perturbed_case_Q5iv.dat');

% cartesian coordinates
Juice_unperturbed_Gframe               = CartesianCoordinatesQ2(:,2:4);
Juice_unperturbed_Gframe_Jpropagation  = Juice_unperturbed_Jframe(:,2:4) - Ganymede_unperturbed_Jframe(:,2:4) ; % juice position in Ganymede refernce, but obtained from the propagation around jupiter
Juice_perturbed_Gframe                 = CartesianCoordinatesQ3(:,2:4);
Juice_perturbed_Gframe_Jpropagation    = Juice_perturbed_Jframe(:,2:4) -  Ganymede_perturbed_Jframe(:,2:4);

delta_r_i_ii_norm   = sqrt(sum((Juice_unperturbed_Gframe - Juice_unperturbed_Gframe_Jpropagation).^2,2));
delta_r_iii_iv_norm = sqrt(sum((Juice_perturbed_Gframe - Juice_perturbed_Gframe_Jpropagation).^2,2));

fig_task5a_effciency=figure('Name','Question 5a difference in position','Position',[50,50,1000,400],'visible',visible);

semilogy(time_days,delta_r_i_ii_norm,'linewidth',2,'DisplayName','unperturbed');
hold on
semilogy(time_days,delta_r_iii_iv_norm,'linewidth',2,'DisplayName','perturbed');
ax=myaxes(gca());

grid on
xlabel('time (days)')
ylabel('displacement (m)')
legend()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ganymede and juice are in the same matrix in this case

% cartesian coordinates
Juice_and_Ganymede_unperturbed_Jframe    = dlmread('OUTPUTFILES/JUICE_cartesianstate_unperturbed_Q6.dat');
Juice_and_Ganymede_perturbed_Jframe      = dlmread('OUTPUTFILES/JUICE_cartesianstate_perturbed_Q6.dat');

acceleration_q6     = dlmread('Acceleration_from_Ganymede_on_Juice_Q6.dat')
fig_task6a_acc=figure('Name','Question 6a difference in acc','Position',[50,50,1000,400]);

plot(time_days,sqrt(sum((acceleration_q6(:,2:end)-acceleration_q2).^2,2)),'linewidth',2,'DisplayName','unperturbed');

xlabel('time (days)')
ylabel('||(a_{Gs})_J - (a_{Gs})_G||')

% [2:7] state of juice position and speed
% [8:13] state of juice position and speed

% cartesian coordinates
Juice_unperturbed_Gframe                   = CartesianCoordinatesQ2(:,2:4);
Juice_perturbed_Gframe                     = CartesianCoordinatesQ3(:,2:4);
Juice_unperturbed_Jframe                   = Juice_and_Ganymede_unperturbed_Jframe(:,2:4);
Juice_perturbed_Jframe                     = Juice_and_Ganymede_perturbed_Jframe(:,2:4);

Ganymede_unperturbed_Jframe_Jpropagation   = Juice_and_Ganymede_unperturbed_Jframe(:,8:10);
Juice_unperturbed_Gframe_Jpropagation      = Juice_unperturbed_Jframe - Ganymede_unperturbed_Jframe_Jpropagation; % juice position in Ganymede refernce, but obtained from the propagation around jupiter


Ganymede_perturbed_Jframe_Jpropagation     = Juice_and_Ganymede_perturbed_Jframe(:,8:10);
Juice_perturbed_Gframe_Jpropagation        = Juice_perturbed_Jframe - Ganymede_perturbed_Jframe_Jpropagation;

delta_r_1  = sqrt(sum((Juice_unperturbed_Gframe   - Juice_unperturbed_Gframe_Jpropagation).^2,2));
delta_r_2  = sqrt(sum((Juice_perturbed_Gframe- Juice_perturbed_Gframe_Jpropagation).^2,2));

fig_task6a_effciency=figure('Name','Question 6a difference in position','Position',[50,50,1000,400],'visible',visible);

semilogy(time_days,delta_r_1,'linewidth',2,'DisplayName','unperturbed');
hold on
semilogy(time_days,delta_r_2,'linewidth',2,'DisplayName','perturbed');
ax=myaxes(gca());

grid on
xlabel('time (days)')
ylabel('displacement (m)')
legend('location','best')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot kartesian state for the two cases

KeplerianState_unperturbed        = dlmread('OUTPUTFILES/JUICE_keplerelement_unperturbed_Q7.dat');
KeplerianState_perturbed          = dlmread('OUTPUTFILES/JUICE_keplerelement_perturbed_Q7.dat');
parameters                        = dlmread('OUTPUTFILES/parameters.txt');
time_days               = (KeplerianState_unperturbed(:,1)-KeplerianState_unperturbed(1,1))/60/60/24; %days

kep_labels = {'\Delta a [m]',' \Delta e','\Delta i [rad]','\Delta \omega [rad]',' \Delta \Omega [rad]','\Delta \theta[rad]'};

fig_task7_unperturbed=figure('Position',[50,50,900,600],'visible',visible);

for jj=1:6
    ax = myaxes(subplot(3,2,jj));
    hold on
    plot(ax,time_days,KeplerianState_unperturbed(1,jj+1)-KeplerianState_unperturbed(:,jj+1));
    xlabel('days');
    ylabel(kep_labels{jj});
end

fig_task7_perturbed=figure('Position',[50,50,900,600],'visible','on');

for jj=1:6
    ax = myaxes(subplot(3,2,jj));
    hold on
    plot(ax,time_days,KeplerianState_perturbed(1,jj+1)-KeplerianState_perturbed(:,jj+1));
    xlabel('days');
    ylabel(kep_labels{jj});
end

fig_task7_trueanomaly=figure('Position',[50,50,900,600],'visible',visible);
ax = myaxes(gca());
hold on
plot(ax,time_days,unwrap(KeplerianState_perturbed(:,7)-KeplerianState_unperturbed(:,7)));
xlabel('days');
ylabel('\Delta \theta');



% parameters [ganymede_j2 ,ganymede_unnormalized_c22,ganymede_reference_radius,jupiter_j2,jupiter_reference_radius]

ganymede_j2               = parameters(1);
ganymede_unnormalized_c22 = parameters(2);
ganymede_reference_radius = parameters(3);
jupiter_j2                = parameters(4);
jupiter_reference_radius  = parameters(5);
jupiter_mu                = parameters(6);


mean_anomaly_unperturbed = sqrt(jupiter_mu./KeplerianState_unperturbed(:,2).^3);
mean_anomaly_perturbed   = sqrt(jupiter_mu./KeplerianState_perturbed(:,2).^3);

delta_mean_anomaly = mean_anomaly_unperturbed - mean_anomaly_perturbed;
semimajor_axis     = mean(KeplerianState_unperturbed(:,2));


% Analytical mean motion change 

dn_j2_jupiter   =   -sqrt(semimajor_axis/jupiter_mu)/2.*  jupiter_mu*1.5 * jupiter_j2               * jupiter_reference_radius.^2./semimajor_axis.^4;
dn_j2_ganymede  =   -sqrt(semimajor_axis/jupiter_mu)/2.*  jupiter_mu*1.5 * ganymede_j2              * ganymede_reference_radius^2./semimajor_axis.^4;
dn_c22_ganymede =   -sqrt(semimajor_axis/jupiter_mu)/2.* -jupiter_mu*9   * ganymede_unnormalized_c22* ganymede_reference_radius.^2./semimajor_axis.^4


dn_sum = dn_j2_jupiter + dn_j2_ganymede + dn_c22_ganymede ;

fig_task7_mean_anomaly=figure('Position',[50,50,900,600]);

ax=subplot(211);
plot(time_days,dn_sum*ones(size(time_days)));
ylim([dn_sum*1.1 dn_sum*0.9]);
xlabel('days')
ylabel('\Delta n [1/s]')
legend('Analytical','location','best')
ax = myaxes(ax);

ax=subplot(212);
plot(time_days,delta_mean_anomaly);

xlabel('days')
ylabel('\Delta n [1/s]')
legend('Numerical')
ax=myaxes(ax);

%% print the images

if print_images 
    exportgraphics(fig_task1a,fullfile(imagesFolder,'task1a.eps'))
    exportgraphics(fig_task2a,fullfile(imagesFolder,'task2a.eps'))
    exportgraphics(Ganymede_figure,fullfile(imagesFolder,'task2a_ganymede_image.eps'))
    exportgraphics(fig_task3ai_ii,fullfile(imagesFolder,'task3ii_i.eps'))
    exportgraphics(fig_task3d,fullfile(imagesFolder,'task3d.eps'))
    exportgraphics(fig_task3kepler,fullfile(imagesFolder,'task3d_kepler.eps'))
    exportgraphics(fig_task4a,fullfile(imagesFolder,'task4a.eps'))
    exportgraphics(fig_task4b_effciency,fullfile(imagesFolder,'task4a_efficiency.eps'))
    exportgraphics(fig_task4b_com,fullfile(imagesFolder,'task4a_componenets.eps'))
    exportgraphics(fig_task5a_effciency,fullfile(imagesFolder,'task5a_j_vs_G_.eps'))
    exportgraphics(fig_task6a_effciency,fullfile(imagesFolder,'task6a_j_vs_G_.eps'))
    exportgraphics(fig_task6a_acc,fullfile(imagesFolder,'task6a_acc.eps'))
    exportgraphics(fig_task7_mean_anomaly,fullfile(imagesFolder,'task7.eps'))
    exportgraphics(fig_task7_perturbed,fullfile(imagesFolder,'task7_kepler.eps'))
end

%%



