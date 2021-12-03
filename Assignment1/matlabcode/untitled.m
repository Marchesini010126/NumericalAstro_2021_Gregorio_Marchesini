%% the final destination
close all
clear all
theta  = 0:0.01:2*pi;
theta1 = 0:0.01:2*pi;
figure

mu_sun  = 1.32*10^20;
gamma1  = 40*pi/180;
gamma   = 220*pi/180;
counter =1;

for jj = theta
   
hold on
gamma   = gamma *pi/180;

rothh   = cos(2*jj) *  [cos(gamma),-sin(gamma),0;
                       sin(gamma),cos(gamma),0;
                         0        ,      0   ,1];

rotkk =  [ 1         ,      0   ,0;
           0, cos(gamma1),-sin(gamma1);
           0,sin(gamma1),cos(gamma1)];
       
       
pos_sun      = [1e11,0,0];
acc_ganymede = mu_sun*[-1,0,0]./norm(pos_sun)^2;
pos_juice    = (1e6+1e4*cos(2*jj)).*[cos(jj),0,sin(jj)];

% figure
% scatter3(pos_juice(:,1),pos_juice(:,2),pos_juice(:,3))
% xlabel('x')
% ylabel('y')
% zlabel('z')
% axis equal
pos_juice    = (rothh*rotkk*pos_juice')';
juice_sun    = (pos_sun-pos_juice);
juice_sun_n  = sqrt(sum(juice_sun.^2,2));
acc_juice    = mu_sun*juice_sun./juice_sun_n.^3;
acc_rel(counter)      = sqrt(sum((acc_ganymede-acc_juice).^2,2));
counter = counter+1
end
semilogy(acc_rel,'DisplayName',num2str(gamma*180/pi))
% figure
% scatter3(pos_juice(:,1),pos_juice(:,2),pos_juice(:,3))
% xlabel('x')
% ylabel('y')
% zlabel('z')
% axis equal