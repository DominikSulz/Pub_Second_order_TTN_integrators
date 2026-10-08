clear all; close all; clc
addpath('../rank_adaptive_integrator_for_TTN')
addpath('../Second_order_paralllel_BUG')

% number particles
d = 8;
% parameters of the model
Omega = 1;
% time step size and final time
T_end = 1;
dt = 0.01;
% rank of initial data at bottom layer
r = 2;
% bond dimension operator
r_op_max = 10;
r_op_min = 2;

% for rank-adaptive integrator
tol = 10^-8;
r_min = 2;
r_max = 16;

% constant for step rejection
const = 10;

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
% X = rounding(X,tau);
% [X,tau] = init_spin_all_dim_rank2(r,2,d);

% test random inital data
% [X,tau] = init_spin_all_dim_diff_rank_rand(r,2,d);
% X{end} = X{end}/Mat0Mat0(X,X);
% X = rounding(X,tau);
% X = truncate(X,10^-14,100,2);

% [X,tau] = init_spin_all_dim_same_rank(r,2,d);
% X = rounding(X,tau);

% [X,tau] = init_spin_up_and_down(r,2,d);
% X = truncate(X,10^-12,100,2);    

% initial data Tucker tensor
% [X,tau] = init_Tucker(d);

% create cell array for Hamiltonian
B = linearisation_Ising(Omega*sx,sz,d);

% make operator of Ising model in TTN representation -> change that to HSS construction
A = make_operator(X,B,tau,2*ones(d,1));
A{end} = -A{end};
A{end} = -1i*A{end};
A = rounding(A,tau);
A = truncate(A,10^-14,r_op_max,r_op_min);

% [X,~,~] = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X,@F_Ising,0,dt,A,d,r_min);
% X = rounding(X,tau);
% X = truncate(X,tol,r_max,r_min);

% make operator magnetisation 
B2 = cell(d,d);
for ii=1:d
    B2{ii,ii} = sz;
end
Mag = make_operator(X,B2,tau,2*ones(d,1));

% time evolution
time = [];
obs_sz = [];
obs_sz_ad = [];
obs_sz_2nd = [];
obs_sz_BUG = [];

tmp = F_Ising(0,X,A,d);
en = -1i*Mat0Mat0(X,tmp);
nn = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_ad = -1i*Mat0Mat0(X,tmp);
nn_ad = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_2nd = -1i*Mat0Mat0(X,tmp);
nn_2nd = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_BUG = -1i*Mat0Mat0(X,tmp);
nn_BUG = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

r_max_pa = max_rank(X);
r_max_ad = max_rank(X);
r_max_2nd = max_rank(X);
r_max_BUG = max_rank(X);

X_start = X;
X_start_ad = X;
X_start_2nd = X;
X_start_BUG = X;


for it=1:Iter
    t0 = (it-1)*dt;
    t1 = it*dt;
    time(it) = t1;
    
%     % without step rejection
    [X_new,~,~] = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
    % X_new = rounding(X_new,tau);
    X_new = truncate(X_new,tol,r_max,r_min);
    
%     % with step rejection (naive way)
    % rej = 1;
    % while rej == 1
    %     [X_new,U_tilde,~] = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
    %     X_new = truncate(X_new,tol,r_max,r_min);
    %     rej_rk = rejection_check(X_start,X_new);
    %     U_tilde_SR = U_tilde_TTN(X_start,X_new);
    %     rej_eta = eta_check(X_new,U_tilde,U_tilde_SR,F_Ising(t0,X_new,A,d),const,dt,tol);
    %     if (rej_rk == 1) || (rej_eta == 1)
    %         X_start = augment_zero(X_start,X_new); % Kann evtl. paralleler BUG nicht mit 0-Zeilen umghen?
    %         X_start = orthogonalize(X_start);
    %     else
    %         rej = 0;
    %     end
    % end
    
    %     % time integration 2nd order parallel integrator
    [U0_hat,M_save,CF_save] = pre_augment(tau,X_start_2nd,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1);
    [Q_save,Q_hat_save,S_save] = pre_construct_Q(tau,X_start_2nd,U0_hat,M_save,CF_save,@F_Ising,t0,t1,A,d,r_min,1);
    % X_new_2nd = Second_order_parallel_for_TTN(tau,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save,M_save);
    X_new_2nd = Second_order_rank_adaptive_for_TTN(tau,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save,M_save);
    % X_new_2nd = rounding(X_new_2nd,tau);
    X_new_2nd = truncate(X_new_2nd,tol,r_max,r_min);
    
    % rank-adaptive
    X_new_ad = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X_start_ad,@F_Ising,t0,t1,A,d,r_min);
    X_new_ad = truncate(X_new_ad,tol,r_max,r_min);
    
%     % BUG integrator
%     X_new_ad = TTN_integrator_complex_nonglobal_spin(tau,X_start_ad,@F_Ising,t0,t1,A,d);
    
    % norm and energy 
    en(it+1) = -1i*Mat0Mat0(X_new,F_Ising(0,X_new,A,d)); % times i, as we only want the Hamiltonian
    nn(it+1) = sqrt(abs(Mat0Mat0(X_new,X_new)));
    
    en_ad(it+1) = -1i*Mat0Mat0(X_new_ad,F_Ising(0,X_new_ad,A,d)); % times i, as we only want the Hamiltonian
    nn_ad(it+1) = sqrt(abs(Mat0Mat0(X_new_ad,X_new_ad)));

    en_2nd(it+1) = -1i*Mat0Mat0(X_new_ad,F_Ising(0,X_new_2nd,A,d)); % times i, as we only want the Hamiltonian
    nn_2nd(it+1) = sqrt(abs(Mat0Mat0(X_new_ad,X_new_2nd)));
    
%     % renormalisation
%     X_new{end} = X_new{end}/sqrt(abs(Mat0Mat0(X_new,X_new)));
%     sqrt(abs(Mat0Mat0(X_new,X_new)))
    
%     en_BUG(it+1) = -1i*Mat0Mat0(X_new_BUG,F_Ising(0,X_new_BUG,A,d)); % times i, as we only want the Hamiltonian
%     nn_BUG(it+1) = sqrt(abs(Mat0Mat0(X_new_BUG,X_new_BUG)));
    
    % setting for next time step
    X_start = X_new; 
    X_start_ad = X_new_ad;
    X_start_2nd = X_new_2nd;
%     X_start_BUG = X_new_BUG;
    
    % compute magnetisation in z-direction
    tmp = apply_operator_nonglobal(X_new,Mag,d);
    obs_sz(it) = (1/d)*Mat0Mat0(X_new,tmp);
    
    tmp = apply_operator_nonglobal(X_new_ad,Mag,d);
    obs_sz_ad(it) = (1/d)*Mat0Mat0(X_new_ad,tmp);

    tmp = apply_operator_nonglobal(X_new_2nd,Mag,d);
    obs_sz_2nd(it) = (1/d)*Mat0Mat0(X_new_2nd,tmp);
    
%     tmp = apply_operator_nonglobal(X_new_BUG,Mag,d);
%     obs_sz_BUG(it) = (1/d)*Mat0Mat0(X_new_BUG,tmp);
    
    % max rank
    r_max_pa(it+1) = max_rank(X_new);
    r_max_ad(it+1) = max_rank(X_new_ad);
    r_max_2nd(it+1) = max_rank(X_new_2nd);
%     r_max_BUG(it+1) = max_rank(X_new_BUG);

    % error at time t1
    if d<=8
        sol = ref_sol_spin_systems(X,0,t1,B,d,2,1);
        m = length(size(X{end})) - 1;
        T1 = double(tenmat(full_tensor(X_new),m+1,1:(m))).';
        T2 = double(tenmat(full_tensor(X_new_ad),m+1,1:(m))).';
        T3 = double(tenmat(full_tensor(X_new_2nd),m+1,1:(m))).';
        err(it) = norm(T1 - sol);
        err_ad(it) = norm(T2 - sol);
        err_2nd(it) = norm(T3 - sol);
    end
    
    it
    
end


figure(1)
plot(time,real(obs_sz),'Linewidth',1.5)
hold on
plot(time,real(obs_sz_ad),'-.','Linewidth',1.5)
plot(time,real(obs_sz_2nd),'-.','Linewidth',1.5)
% plot(time,real(obs_sz_BUG),':','Linewidth',1.5)
xlabel('Time')
title('Magnetization in z-direction')
legend('Parallel Integrator','Rank-adaptive BUG','2nd order parallel BUG')

figure(2)
plot([0 time],abs(nn),'Linewidth',2)
hold on
plot([0 time],abs(nn_ad),'-.','Linewidth',2)
plot([0 time],abs(nn_2nd),'-.','Linewidth',2)
% plot([0 time],abs(nn_BUG),':','Linewidth',2)
xlabel('Time')
title('Norm')
legend('Parallel Integrator','Rank-adaptive BUG','2nd order parallel BUG')

figure(3)
plot([0 time],abs(en),'Linewidth',2)
hold on
plot([0 time],abs(en_ad),'-.','Linewidth',2)
plot([0 time],abs(en_2nd),'-.','Linewidth',2)
% plot([0 time],abs(en_BUG),':','Linewidth',2)
xlabel('Time')
title('Energy')
legend('Parallel Integrator','Rank-adaptive BUG','2nd order parallel BUG')

figure(4) 
plot([0 time],r_max_pa,'Linewidth',2)
hold on
plot([0 time],r_max_ad,'-.','Linewidth',2)
plot([0 time],r_max_2nd,':','Linewidth',2)
xlabel('Time')
title('Max. ranks')
legend('Parallel Integrator','Rank-adaptive BUG','2nd order parallel BUG')

figure(5)
semilogy(dt:dt:T_end,err.','Linewidth',2)
hold on
semilogy(dt:dt:T_end,err_ad.','-.','Linewidth',2)
semilogy(dt:dt:T_end,err_2nd.','-.','Linewidth',2)
xlabel('Time')
title('Error over time')
legend('Parallel Integrator','Rank-adaptive BUG','2nd order parallel BUG')