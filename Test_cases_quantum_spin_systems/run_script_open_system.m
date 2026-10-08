clear all; close all; clc
addpath('../rank_adaptive_integrator_for_TTN')
addpath('../Second_order_paralllel_BUG')

% number particles
d = 2;

% parameters of the model
Omega = 0.4;
Delta = -2;
gamma = 1;
alpha = 0;
c_alpha=sum((1:1:d).^(-alpha));
V = 2/c_alpha;

% time step size and final time
T_end = 1;
dt = 0.025;

r_vec = [10 10 10 6 4]; % for L=16
% r_vec = [10 10 10 10 6 4]; % for L=32

% bond dimension operator
r_op_max = 20;
r_op_min = 4;

% for rank-adaptive integrator
tol = 10^-8;
r_min = 4;
r_max = 10;
r = 4;

% constant for step rejection
const = 10;

%%% time-step
Iter=T_end/dt;

% initialisations
sx=[0,1;1,0];  %% Pauli Matrix x
sy=[0,-1i;1i,0]; %% Pauli Matrix y
sz=[1,0;0,-1]; %% Pauli Matrix z
n=[1,0;0,0];  %% Projector onto the excited state Pu=(sz+id)/2;
id=[1,0;0,1];  %% Identity for the single spin
J = [0,0;1,0];

density = zeros(1,d);

% % % create initial data binary tree
[X,tau] = init_spin_all_dim_diff_rank(r,2,d);

% [X,tau] = init_spin_all_dim_same_rank(r,2,d);
% X = rounding(X,tau);

% [X,tau] = init_spin_up_and_down(r,2,d);
% X = truncate(X,10^-12,100,2);  

% open
[X1,tau] = init_spin_all_dim_same_rank_open(r,2,d,1);
[X2,tau] = init_spin_all_dim_same_rank_open(r,2,d,0);
X = Add_TTN(X1,X2,tau);
X = truncate(X,10^-12,100,2);

% initial data Tucker tensor
% [X,tau] = init_Tucker(d);

flat_state = obs_state(d,r(end));

% create cell array for Hamiltonian
B = linearisation_long_range(d,J,sx,n,V,Delta,Omega,gamma,alpha);

% make operator 
A = make_operator(X,B,tau,4*ones(d,1));
A = rounding(A,tau);
A = truncate(A,10^-8,r_op_max,r_op_min);


% time evolution
time = [];
obs_sz = [];
obs_sz_ad = [];

tmp = F_Ising(0,X,A,d);
en = -1i*Mat0Mat0(X,tmp);
nn = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_ad = -1i*Mat0Mat0(X,tmp);
nn_ad = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

r_max_pa = max_rank(X);
r_max_ad = max_rank(X);

X_start = X;
X_start_ad = X;

tic
for it=1:Iter
    t0 = (it-1)*dt;
    t1 = it*dt;
    time(it) = t1;
    
%     % without step rejection
    [X_new,~,~] = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
    % X_new = rounding(X_new,tau);
    X_new = truncate(X_new,tol,r_max,r_min);
    
    % rank-adaptive
    X_new_ad = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X_start_ad,@F_Ising,t0,t1,A,d,r_min);
    X_new_ad = truncate(X_new_ad,tol,r_max,r_min);

    % setting for next time step
    X_start = X_new; 
    X_start_ad = X_new_ad;
%     X_start_BUG = X_new_BUG;

    % max rank
    r_max_pa(it+1) = max_rank(X_new);
    r_max_ad(it+1) = max_rank(X_new_ad);

    % error at time t1
    if d<=4
        sol = ref_sol_spin_systems(X,0,t1,B,d,4);
        m = length(size(X{end})) - 1;
        T1 = double(tenmat(full_tensor(X_new),m+1,1:(m))).';
        T2 = double(tenmat(full_tensor(X_new_ad),m+1,1:(m))).';
        err(it) = norm(T1 - sol);
        err_ad(it) = norm(T2 - sol);
    end
    
    it
    
end
total_time = toc;

figure(1)
semilogy(dt:dt:T_end,err.','Linewidth',2)
hold on
semilogy(dt:dt:T_end,err_ad.','-.','Linewidth',2)
xlabel('Time')
title('Error over time')
legend('Parallel Integrator','Rank-adaptive BUG')

figure(2) 
plot([0 time],r_max_pa,'Linewidth',2)
hold on
plot([0 time],r_max_ad,'-.','Linewidth',2)
xlabel('Time')
title('Max. ranks')
legend('Parallel Integrator','Rank-adaptive BUG')

