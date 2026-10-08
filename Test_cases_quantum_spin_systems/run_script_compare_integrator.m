clear all; close all; clc;

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
r_op_max = 4;
r_op_min = 2;

% for rank-adaptive integrator
tol = 10^-8;
r_min = 2;
r_max = 4;

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

% make operator magnetisation 
B2 = cell(d,d);
for ii=1:d
    B2{ii,ii} = sz;
end
Mag = make_operator(X,B2,tau,2*ones(d,1));

% time evolution
time = [];
obs_sz_2nd = [];
obs_sz_ad = [];
obs_sz_2nd = [];

tmp = F_Ising(0,X,A,d);
en_2nd = -1i*Mat0Mat0(X,tmp);
nn_2nd = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_ad = -1i*Mat0Mat0(X,tmp);
nn_ad = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_pa = -1i*Mat0Mat0(X,tmp);
nn_pa = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

r_max_pa = max_rank(X);
r_max_ad = max_rank(X);
r_max_2nd = max_rank(X);

X_start_pa = X;
X_start_ad = X;
X_start_2nd = X;

tic
for it=1:Iter
    t0 = (it-1)*dt;
    t1 = it*dt;
    time(it) = t1;
    
     % second order parallel
    [U0_hat,M_save,CF_save] = pre_augment(tau,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1);
    [Q_save,Q_hat_save,S_save] = pre_construct_Q(tau,X_start_2nd,U0_hat,M_save,CF_save,@F_Ising,t0,t1,A,d,r_min,1);
    X_new_2nd = Second_order_parallel_for_TTN(tau,X_start_2nd,@F_Ising,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save);
    X_new_2nd = rounding(X_new_2nd,tau);
    X_new_2nd = truncate(X_new_2nd,tol,r_max,r_min);
    
    % rank-adaptive
    X_new_ad = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X_start_ad,@F_Ising,t0,t1,A,d,r_min);
    X_new_ad = truncate(X_new_ad,tol,r_max,r_min);
    
    % parallel BUG integrator with new Q
    X_new_pa = TTN_integrator_complex_parallel_nonglobal_newQ(tau,X_start_pa,@F_Ising,t0,t1,A,d,r_min);
    X_new_pa = rounding(X_new_pa,tau);
    X_new_pa = truncate(X_new_pa,tol,r_max,r_min);
    
    % norm and energy 
    en_2nd(it+1) = -1i*Mat0Mat0(X_new_2nd,F_Ising(0,X_new_2nd,A,d)); % times i, as we only want the Hamiltonian
    nn_2nd(it+1) = sqrt(abs(Mat0Mat0(X_new_2nd,X_new_2nd)));
    
    en_ad(it+1) = -1i*Mat0Mat0(X_new_ad,F_Ising(0,X_new_ad,A,d)); % times i, as we only want the Hamiltonian
    nn_ad(it+1) = sqrt(abs(Mat0Mat0(X_new_ad,X_new_ad)));

    en_pa(it+1) = -1i*Mat0Mat0(X_new_pa,F_Ising(0,X_new_pa,A,d)); % times i, as we only want the Hamiltonian
    nn_pa(it+1) = sqrt(abs(Mat0Mat0(X_new_pa,X_new_pa)));

    % setting for next time step
    X_start_2nd = X_new_2nd; 
    X_start_ad = X_new_ad;
    X_start_pa = X_new_pa;
    
    % compute magnetisation in z-direction
    tmp_2nd = apply_operator_nonglobal(X_new_2nd,Mag,d);
    obs_sz_2nd(it) = (1/d)*Mat0Mat0(X_new_2nd,tmp);
    
    tmp_ad = apply_operator_nonglobal(X_new_ad,Mag,d);
    obs_sz_ad(it) = (1/d)*Mat0Mat0(X_new_ad,tmp);
    
    tmp = apply_operator_nonglobal(X_new_pa,Mag,d);
    obs_sz_pa(it) = (1/d)*Mat0Mat0(X_new_pa,tmp);


    % compare to ref sol
    sol = ref_sol(X,B,d,t1);
    [err_2nd(it),err_ad(it),err_pa(it)] = compare_to_ref(X_new_2nd,X_start_ad,X_start_pa,sol);
    
    % max rank
    r_max_2nd(it+1) = max_rank(X_new_2nd);
    r_max_ad(it+1) = max_rank(X_new_ad);
    r_max_pa(it+1) = max_rank(X_new_pa);
    
    it
    
end

% figure(1)
% plot(time,obs_ref,'k','Linewidth',1.5)
% hold on
% plot(time,real(obs_sz_2nd),'Linewidth',1.5)
% plot(time,real(obs_sz_ad),'-.','Linewidth',1.5)
% plot(time,real(obs_sz_pa),':','Linewidth',1.5)
% xlabel('Time')
% title('Magnetization in z-direction')
% legend('Exact reference solution','2nd order parallel Integrator','Rank-adaptive BUG','Parallel Integrator')

figure(2)
plot([0 time],abs(nn_2nd),'Linewidth',2)
hold on
plot([0 time],abs(nn_ad),'-.','Linewidth',2)
plot([0 time],abs(nn_pa),':','Linewidth',2)
xlabel('Time')
title('Norm')
legend('2nd order parallel Integrator','Rank-adaptive BUG','Parallel Integrator')

figure(3)
plot([0 time],abs(en_2nd),'Linewidth',2)
hold on
plot([0 time],abs(en_ad),'-.','Linewidth',2)
plot([0 time],abs(en_pa),':','Linewidth',2)
xlabel('Time')
title('Energy')
legend('2nd order parallel Integrator','Rank-adaptive BUG','Parallel Integrator')

figure(4) 
plot([0 time],r_max_pa,'Linewidth',2)
hold on
plot([0 time],r_max_ad,'-.','Linewidth',2)
plot([0 time],r_max_pa,':','Linewidth',2)
xlabel('Time')
title('Max. ranks')
legend('2nd order parallel Integrator','Rank-adaptive BUG','Parallel Integrator')

% figure(5)
% semilogy(time,abs(obs_ref - obs_sz_2nd),'Linewidth',2)
% hold on
% semilogy(time,abs(obs_ref - obs_sz_ad),'Linewidth',2)
% semilogy(time,abs(obs_ref - obs_sz_pa),'Linewidth',2)
% legend('Error 2nd order parallel Integrator','Error Rank-adaptive BUG','Error Parallel Integrator')

figure(6)
semilogy(time,err_2nd,'Linewidth',2)
hold on
semilogy(time,err_ad,'Linewidth',2)
semilogy(time,err_pa,'Linewidth',2)
legend('Error 2nd order parallel Integrator','Error Rank-adaptive BUG','Error Parallel Integrator')


function [psi] = ref_sol(X0,B,L,t)

psi0 = double(full_tensor(X0)).';
psi0 = psi0(:);
% psi0 = double(tenmat(psi0,L+1,1:L)).';

H_sys = - superkron(L,B);

% H_sys = double(full_tensor(A));

psi=expm(-1i*t*H_sys)*psi0;

end


function [err1,err2,err3] = compare_to_ref(X1,X2,X3,sol)


T = double(full_tensor(X1));
mat1 = T(:);

T = double(full_tensor(X2));
mat2 = T(:);

T = double(full_tensor(X3));
mat3 = T(:);

err1 = norm(sol - mat1(:));
err2 = norm(sol - mat2(:));
err3 = norm(sol - mat3(:));

end


% 
% 
% function [obs_exact,psi] = ref_sol(d,B,dt,T_end)
% 
% %%%% dimension
% L=d;
% 
% FinalTime=T_end;
% Iter=FinalTime/dt;
% 
% 
% sx=[0,1;1,0];
% sy=[0,-i;i,0];
% sz=[1,0;0,-1];
% 
% iden=[1 0;0 1];
% 
% H_sys = superkron(L,B);
% 
% 
% 
% 
% Proj1=[1 0;0 0];
% Proj2=[0 0;0 1];
% 
% tmp2=diag(sparse(ones(2^(L-1),1)));
% X{1,1}=kron(sx,tmp2);
% X{L,1}=kron(tmp2,sx);
% 
% Z{1,1}=kron(sz,tmp2);
% Z{L,1}=kron(tmp2,sz);
% 
% 
% for i1=2:L-1
%    tmp1=diag(sparse(ones(2^(i1-1),1)));
%    tmp2=diag(sparse(ones(2^(L-i1),1)));
%    X{i1,1}=kron(kron(tmp1,sx),tmp2);
%    Z{i1,1}=kron(kron(tmp1,sz),tmp2);
% end
% 
% Mag=0*X{1,1};
% H_X=0*X{1,1};
% 
% for i1=1:L
%     Mag=Mag+Z{i1,1};
%     H_X=H_X+X{i1,1};
% end
% 
% % H_sys=-h*H_X-H_NN;
% 
% psi=zeros(2^L,1);
% psi(1,1)=1;
% 
% tic
% Vt=expm(-1i*dt*H_sys); % Vt=expm(-1i*dt*H_sys);
% toc
% 
% for tst=1:Iter
%     time(tst)=tst*dt;
%     psi=Vt*psi;
%     psi=psi/norm(psi);
% 
%     p1(tst)=dot(psi,Mag*psi)/L;
%     energy(tst)=dot(psi,H_sys*psi)/L;
% 
%     mag(tst)=dot(psi,Mag*psi)/L;
% end
% 
% 
% %%%% diagonalization
% 
% Ground_State_Energy=min(real(eig(H_sys)))/L;
% %%%%% diagonalization of H_eff
% [v2,d2]=eig(full(H_sys));
% [x2,y2]=sort(diag(d2),'ascend');
% 
% 
%  %%% positive eigenvalue
%  ind_p=y2(end);
% 
% 
%  norm(H_sys*v2(:,ind_p)-d2(ind_p,ind_p)*v2(:,ind_p));
%  Ground_state=v2(:,ind_p);
%  Ground_state=Ground_state/norm(Ground_state);
% 
% 
%  Mag_Ground_state=dot(Ground_state,Mag*Ground_state)/L;
% 
% 
% obs_exact = mag;
% 
% end



function [M] = superkron(L,B)
% B is a cell array
[m,n] = size(B);

M = sparse(2^L,2^L);
for jj=1:m
    tmp = 1;
    for ii=n:-1:1
        if isempty(B{jj,ii}) == 1
            tmp = kron(tmp,speye(2,2));
        else
            tmp = kron(tmp,B{jj,ii});
        end
    end
    M = M + tmp;
end

end
