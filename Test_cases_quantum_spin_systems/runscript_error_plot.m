clear all; close all; clc;

addpath('../rank_adaptive_integrator_for_TTN')
addpath('../Second_order_paralllel_BUG')

% number particles
d = 8;
% parameters of the model
Omega = 1;
% time step size and final time
T_end = 1;
dt_save = 0.01 ./ 2.^(0:8); % [0.025 0.01 0.005 0.0025 0.001 0.0005 0.00025]; 

ref_exact = true;

% rank of initial data at bottom layer
r = 2;
% bond dimension operator
r_op_max = 10;
r_op_min = 2;

% for rank-adaptive integrator
tol = 0*10^-14;
r_min = 2;
r_max = 10;

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

% time evolution
time = [];
obs_sz = [];
obs_sz_ad = [];
obs_sz_BUG = [];

% compute the reference solution via augmented BUG
if ref_exact == 0
    dt_ref = min(dt_save)/2;
    dt_ref = 1.5e-5;
    tol = 0*0.1*dt_ref^3;
    [sol,sol_ad] = ref_sol_spin_systems_aug_BUG(X,tau,dt_ref,0,T_end,A,@F_Ising,d,tol,r_min,r_max);
    sol = double(full_tensor(sol));
    sol = sol(:);
    sol_ad = double(full_tensor(sol_ad));
    sol_ad = sol_ad(:);
end

for kk=1:length(dt_save)
    dt = dt_save(kk);
    tol = 0.1*dt^3;
    % tol = 10^-14;
    %%% time-step
    Iter=T_end/dt;
    % X_start = X;
    X_start_ad = X;
    % X_start_BUG = X;
    X_start_2nd = X;

    tic
    for it=1:Iter
        t0 = (it-1)*dt;
        t1 = it*dt;
        time(it) = t1;
        
        % time integration with parallel integrator
        % X_new = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
        % X_new = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
        % X_new = truncate(X_new,tol,r_max,r_min);

        % % rank-adaptive
        % X_new_ad = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X_start_ad,@F_Ising,t0,t1,A,d,r_min);
        % X_new_ad = truncate(X_new_ad,tol,r_max,r_min);

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
        
        % % BUG integrator
        % X_new_BUG = TTN_integrator_complex_nonglobal_spin(tau,X_start_BUG,@F_Ising,t0,t1,A,d);
        
        % setting for next time step
        % X_start = X_new;
        X_start_ad = X_new_ad;
        % X_start_BUG = X_new_BUG;
        X_start_2nd = X_new_2nd;
        
    end

    if (d<=10) && (ref_exact==1)
        if longrange 
           sol = ref_sol_spin_systems(X,0,T_end,B,d,2,1);
        else 
           [obs_ref,sol] = ref_sol(d,Omega,dt,T_end);
        end

        sol_ra_BUG = double(full_tensor(X_new_ad));
        % sol_par = double(full_tensor(X_new));
        sol_2nd = double(full_tensor(X_new_2nd));
        err_ad(kk) = norm(sol - sol_ra_BUG(:));
        err_2nd(kk) = norm(sol - sol_2nd(:));
        % err_BUG(kk) = norm(sol - sol_BUG(:));
    elseif ref_exact == 0
        sol_ra_BUG = double(full_tensor(X_new_ad));
        % sol_par = double(full_tensor(X_new));
        sol_2nd = double(full_tensor(X_new_2nd));
        err_ad(kk) = norm(sol_ad - sol_ra_BUG(:));
        err_2nd(kk) = norm(sol - sol_2nd(:));
        % err_BUG(kk) = norm(sol - sol_BUG(:));
    end

    kk
    
end




figure(1)
loglog(dt_save,err_2nd,'-o','Linewidth',3)
hold on
loglog(dt_save,err_ad,'-*','Linewidth',3)
% loglog(dt_save,err_par,'-.','Linewidth',2.5)
% loglog(dt_save,dt_save,'-.','Linewidth',2)
C = 1.5*err_2nd(end) / dt_save(end)^2;
loglog(dt_save,0.003*C*dt_save.^2,'--','Linewidth',2)
grid on 
set(gca,'YMinorGrid','off', 'FontSize', 20)
% title('Error','FontSize', 14)
xlabel('Time step size','FontSize', 20)
ylabel('Error at T=1','FontSize', 20)
legend('Second-order parallel BUG','Second-order augmented BUG','h^2','FontSize', 20)
title('Convergence','FontSize', 20)
xlabel('Time step size h','FontSize', 20)
xlim([min(dt_save),max(dt_save)])


function [obs_exact,psi] = ref_sol(d,Omega,dt,T_end)

%%%% dimension
L=d;
h=Omega;

FinalTime=T_end;
Iter=FinalTime/dt;


sx=[0,1;1,0];
sy=[0,-i;i,0];
sz=[1,0;0,-1];

iden=[1 0;0 1];




Proj1=[1 0;0 0];
Proj2=[0 0;0 1];

tmp2=diag(sparse(ones(2^(L-1),1)));
X{1,1}=kron(sx,tmp2);
X{L,1}=kron(tmp2,sx);

Z{1,1}=kron(sz,tmp2);
Z{L,1}=kron(tmp2,sz);


for i1=2:L-1
    tmp1=diag(sparse(ones(2^(i1-1),1)));
    tmp2=diag(sparse(ones(2^(L-i1),1)));
    X{i1,1}=kron(kron(tmp1,sx),tmp2);
    Z{i1,1}=kron(kron(tmp1,sz),tmp2);
end

Mag=0*X{1,1};
H_X=0*X{1,1};

for i1=1:L
    Mag=Mag+Z{i1,1};
    H_X=H_X+X{i1,1};
end

H_NN=0*X{1,1};
for i1=1:L-1
    H_NN=H_NN+Z{i1,1}*Z{i1+1,1};
end

H_sys=-h*H_X-H_NN;

psi=zeros(2^L,1);
psi(1,1)=1;

tic
Vt=expm(-1i*dt*H_sys); % Vt=expm(-1i*dt*H_sys);
toc

for tst=1:Iter
    time(tst)=tst*dt;
    psi=Vt*psi;
    psi=psi/norm(psi);
    
    p1(tst)=dot(psi,Mag*psi)/L;
    energy(tst)=dot(psi,H_sys*psi)/L;
    
    mag(tst)=dot(psi,Mag*psi)/L;
end


%%%% diagonalization

Ground_State_Energy=min(real(eig(H_sys)))/L;
%%%%% diagonalization of H_eff
[v2,d2]=eig(full(H_sys));
[x2,y2]=sort(diag(d2),'ascend');


%%% positive eigenvalue
ind_p=y2(end);


norm(H_sys*v2(:,ind_p)-d2(ind_p,ind_p)*v2(:,ind_p));
Ground_state=v2(:,ind_p);
Ground_state=Ground_state/norm(Ground_state);


Mag_Ground_state=dot(Ground_state,Mag*Ground_state)/L;


obs_exact = mag;

end