clear all; close all; clc;

addpath('../rank_adaptive_integrator_for_TTN')
addpath('../Second_order_paralllel_BUG')

% number particles
d = 4;
% parameters of the model
Omega = 1;
% time step size and final time
T_end = 5;
dt = 0.01;

% rank of initial data at bottom layer
r = 2;
% bond dimension operator
r_op_max = 10;
r_op_min = 2;

% for rank-adaptive integrator
tol = 10^-8;
r_min = 2;
r_max = 5;

%%% time-step
Iter=T_end/dt;

% initialisations
sx=[0,1;1,0];  %% Pauli Matrix x
sy=[0,-1i;1i,0]; %% Pauli Matrix y
sz=[1,0;0,-1]; %% Pauli Matrix z

% % % create initial data binary tree
[X,tau] = init_spin_all_dim_diff_rank(r,2,d);
% [X_test,tau_test] = init_spin_all_dim_diff_rank1(1,d);
% [X,tau] = init_spin_all_dim_diff_rank2(d);
X = rounding(X,tau);
X = truncate(X,10^-12,r_max,r_min);

% initial data Tucker tensor
% [X,tau] = init_Tucker(d);

% create cell array for Hamiltonian
longrange = false;
B = linearisation_Ising(Omega*sx,sz,d);
% make operator of Ising model in TTN representation -> change that to HSS construction
A = make_operator(X,B,tau,2*ones(d,1));
A{end} = -A{end};
A{end} = -1i*A{end};
A = rounding(A,tau);
A = truncate(A,10^-14,r_op_max,r_op_min);

% % % long-range Hamiltonian
% longrange = true;
% Omega = 1;
% V = 2;
% Delta = -0.5;
% alpha = 3;
% B = linearisation_long_range_unitary_full(d,sx,[1,0;0,0],V,Delta,Omega,alpha);
% A = make_operator(X,B,tau,2*ones(d,1));
% A{end} = -1i*A{end};
% A = rounding(A,tau);
% A = truncate(A,10^-14,r_op_max,r_op_min);


% make operator magnetisation 
B2 = cell(d,d);
for ii=1:d
    B2{ii,ii} = sz;
end
Mag = make_operator(X,B2,tau,2*ones(d,1));
Mag = rounding(Mag,tau);

% time evolution
time = [];
obs_sz_ad = [];
obs_sz_2nd = [];

tmp = F_Ising(0,X,A,d);
en_2nd = -1i*Mat0Mat0(X,tmp);
nn_2nd = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_ad = -1i*Mat0Mat0(X,tmp);
nn_ad = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

r_max_ad = max_rank(X);
r_max_2nd = max_rank(X);


X_start_2nd = X;
X_start_ad = X;


for it=1:Iter
    t0 = (it-1)*dt;
    t1 = it*dt;
    time(it) = t1;
    
    % second-order augmented BUG
    [U0_hat,M_save,CF_save] = pre_augment(tau,X_start_ad,X_start_ad,@F_Ising,t0,t1,A,d,r_min,1);
    [Q_save,Q_hat_save,S_save] = pre_construct_Q(tau,X_start_ad,U0_hat,M_save,CF_save,@F_Ising,t0,t1,A,d,r_min,1);
    X_new_ad = Second_order_rank_adaptive_for_TTN(tau,X_start_ad,@F_Ising,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save,M_save);
    X_new_ad = truncate(X_new_ad,tol,r_max,r_min);

    % second order parallel BUG
    [U0_hat,M_save,CF_save] = pre_augment(tau,X_start_2nd,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1);
    [Q_save,Q_hat_save,S_save] = pre_construct_Q(tau,X_start_2nd,U0_hat,M_save,CF_save,@F_Ising,t0,t1,A,d,r_min,1);
    X_new_2nd = Second_order_parallel_for_TTN(tau,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save,M_save);
    % X_new_2nd = Second_order_rank_adaptive_for_TTN(tau,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save,M_save);
    % X_new_2nd = rounding(X_new_2nd,tau);
    X_new_2nd = truncate(X_new_2nd,tol,r_max,r_min);

    % norm and energy 
    en_ad(it+1) = -1i*Mat0Mat0(X_new_ad,F_Ising(0,X_new_ad,A,d)); % times i, as we only want the Hamiltonian
    nn_ad(it+1) = sqrt(abs(Mat0Mat0(X_new_ad,X_new_ad)));

    en_2nd(it+1) = -1i*Mat0Mat0(X_new_ad,F_Ising(0,X_new_2nd,A,d)); % times i, as we only want the Hamiltonian
    nn_2nd(it+1) = sqrt(abs(Mat0Mat0(X_new_ad,X_new_2nd)));

    err_norm_ad(it) = abs(nn_ad(it+1) - nn_ad(1));
    err_energy_ad(it) = abs(en_ad(it+1) - en_ad(1));

    err_norm_2nd(it) = abs(nn_2nd(it+1) - nn_2nd(1));
    err_energy_2nd(it) = abs(en_2nd(it+1) - en_2nd(1));
    

    % setting for next time step
    X_start_ad = X_new_ad;
    X_start_2nd = X_new_2nd;
    
    % compute magnetisation in z-direction
    tmp = apply_operator_nonglobal(X_new_ad,Mag,d);
    obs_sz_ad(it) = (1/d)*Mat0Mat0(X_new_ad,tmp);

    tmp = apply_operator_nonglobal(X_new_2nd,Mag,d);
    obs_sz_2nd(it) = (1/d)*Mat0Mat0(X_new_2nd,tmp);
   
    % max rank
    r_max_ad(it+1) = max_rank(X_new_ad);
    r_max_2nd(it+1) = max_rank(X_new_2nd);

    it
    
end


figure(1)
plot(time,real(obs_sz_2nd),'-.','Linewidth',4)
hold on
plot(time,real(obs_sz_ad),'-.','Linewidth',4)
xlabel('Time')
title('Magnetization in z-direction')
legend('2nd order parallel BUG','2nd order augmented BUG')

figure(2)
plot([0 time],abs(nn_2nd),'Linewidth',4)
hold on
plot([0 time],abs(nn_ad),'-.','Linewidth',4)
xlabel('Time')
title('Norm')
legend('2nd order parallel BUG','2nd order augmented BUG')

figure(3)
plot([0 time],abs(en_2nd),'Linewidth',4)
hold on
plot([0 time],abs(en_ad),'-.','Linewidth',4)
% plot([0 time],abs(en_BUG),':','Linewidth',2)
xlabel('Time')
title('Energy')
legend('2nd order parallel BUG','2nd order augmented BUG')

figure(4) 
plot([0 time],r_max_2nd,'Linewidth',4)
hold on
plot([0 time],r_max_ad,'-.','Linewidth',4)
xlabel('Time')
title('Max. ranks')
legend('2nd order parallel BUG','2nd order augmented BUG')

figure(5)
set(gcf,'Color','w');
subplot(1,2,1)
semilogy(time,err_norm_2nd,'Linewidth',4)
hold on
semilogy(time,err_norm_ad,'-.','Linewidth',4)
xlabel('Time','FontSize', 20)
title('Error norm','FontSize', 20)
set(gca,'FontSize',20)
grid on
legend('2nd order parallel BUG','2nd order augmented BUG','FontSize', 20)

subplot(1,2,2)
semilogy(time,err_energy_2nd,'Linewidth',4)
hold on
semilogy(time,err_energy_ad,'-.','Linewidth',4)
xlabel('Time','FontSize', 20)
title('Error energy','FontSize', 20)
set(gca,'FontSize',20)
grid on
legend('2nd order parallel BUG','2nd order augmented BUG','FontSize', 20)



% figure(5)
% semilogy(dt:dt:T_end,err.','Linewidth',2)
% hold on
% semilogy(dt:dt:T_end,err_ad.','-.','Linewidth',2)
% semilogy(dt:dt:T_end,err_2nd.','-.','Linewidth',2)
% xlabel('Time')
% title('Error over time')
% legend('Parallel Integrator','Rank-adaptive BUG','2nd order parallel BUG')