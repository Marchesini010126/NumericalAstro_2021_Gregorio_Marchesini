
close all
mu_sun = 1.327*10^20;
%juice
juice_vec=accelration_types(:,15:17);
juice_vec_n=sqrt(sum(juice_vec.^2,2));
%sun
sun_vec=accelration_types(:,18:20);
sun_vec_n=sqrt(sum(sun_vec.^2,2));

acc_ganymede = sun_vec*mu_sun./sun_vec_n.^3;
acc_ganymede_n = sqrt(sum(acc_ganymede.^2,2));
acc_juice =juice_vec*mu_sun./juice_vec_n.^3;
acc_juice_n=sqrt(sum(acc_juice.^2,2));


acc_rel=+acc_juice -acc_ganymede;


plot(sqrt(sum(acc_juice.^2,2)))
hold on
plot(sqrt(sum(acc_ganymede.^2,2)))

figure
plot(sqrt(sum(acc_rel.^2,2)))

figure

angle = sum(acc_ganymede.*acc_juice,2)./acc_juice_n./acc_ganymede_n
angle = acos(angle)
plot(angle)
