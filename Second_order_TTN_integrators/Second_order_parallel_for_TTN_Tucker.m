function [Y1,U_tilde,U_tilde_SR,C0_bar] = Second_order_parallel_for_TTN_Tucker(tau,Y0,F_tau,t0,t1,A,d,r_min,root,U0_hat,Q_save,Q_hat_save,S_save)
% This function does one time-step with the sedond order parallel  
% integrator for TTNs in a recursuive way.

Y1 = cell(size(Y0));
m = length(Y0) - 2;
U_tilde = cell(1,m+2);
U_tilde_SR = cell(1,m+2);
Ktau_i = cell(1,m);


%% Solving the ODEs - can be parallized 
for i=1:m
    
    Q0_i_hat = Q_hat_save{i};
    Q0_i = Q_save{i};
    S0_i_T = S_save{i};

    % Ki-step
    % Y0_i = Ytau_i(tau{i},Y0{i},S0_i_T.'*Q0_i.'*Q0_i_hat); % initial value for K-step

    %% test construct initial data
    % Ki-step
    tmpM = Mat0Mat0(U0_hat{i},Y0{i});
    Y0_i = Ytau_i(tau{i},U0_hat{i},tmpM*S0_i_T.'*Q0_i.'*Q0_i_hat); % initial value for K-step
    %%%

    F_tau_i = @(t,Y_tau_i,A,d) restriction(...
        F_tau(t,prolongation(Y_tau_i,U0_hat,i,Q0_i_hat),A,d),U0_hat,i,Q0_i_hat);
    
    Y1_i = RK_4_nonglobal(Y0_i,tau{i},F_tau_i,t0,t1,A,d);

    Ktau_i{i} = Y1_i;

    [~,rr] = size(U0_hat{i});
    Y1_i = qr_ordered(U0_hat{i},Y1_i);

    Y1{i} = Y1_i;
    U_tilde{i} = Y1_i;
    U_tilde_SR{i} = Y1_i(:,rr+1:end); % new

end

%% subflow \Psi
% solve the tensor ODE

C0_bar = U0_hat{end};
 
F_ODE = @(C0,F_tau,U0_hat_tau,t0,A,d) func_ODE(C0,F_tau,U0_hat(1:m),t0,A,d);

Y1{end-1} = eye(size(Y0{end-1}));
C1_bar = RK_4_tensor_nonglobal(C0_bar,F_ODE,U0_hat(1:m),F_tau,t0,t1,tau,A,d);

%% Augmentation
rr = size(C1_bar);
ss = rr;
C1_hat = C1_bar;

% check in which dimension and how much we need to augment \bar{C}_tau
aug = ones(1,m);
aug(2,1:m) = zeros(1,m);
for jj=1:m
    if 0 == iscell(Y1{jj})
        [~,s1] = size(Y1{jj});
        [~,s2] = size(U0_hat{jj}); % size(Y0_aug{jj});
        if s1 == s2 
            aug(1,jj) = 0;
        else
            aug(2,jj) = s1 - s2;
        end
    elseif 1 == iscell(Y1{jj})
        s1 = size(Y1{jj}{end});
        s2 = size(C1_hat);
        if s1(end) == s2(jj)
            aug(1,jj) = 0;
        else
            aug(2,jj) = s1(end) - s2(jj);
        end
    end
end

% augment \bar{C}_tau
for jj=1:m
    % compute Ci - only if there is really an augmentation
%     U_tilde{end} = C1_bar;
    if aug(1,jj) ~= 0 

        v = 1:m+1;
        v = v(v~=jj);

        if 1 == iscell(tau{jj})
            Q0_i_hat = Q_hat_save{jj}{end};
        else
            Q0_i_hat = Q_hat_save{jj};
        end
        
        Ci_mat = Mat0Mat0(U_tilde_SR{jj},Ktau_i{jj})*Q0_i_hat.'; % old Ci_mat = Mat0Mat0(U_tilde_SR{jj},Ktau_i{jj})*Q_hat_save{jj}.';
        Ci_size = size(C1_bar);
        Ci_size(jj) = aug(2,jj);
        Ci = tensor(mat2tens(Ci_mat,Ci_size,jj),Ci_size);

        % augmentation 
        tmp = double(tenmat(C1_hat,jj,v));
        
        % if root
        %     vv = 1:m+1;
        %     vv = vv(vv~=jj);
        % else
        %     vv = 1:m;
        %     vv = vv(vv~=jj);
        % end
        mat_Ci = double(tenmat(Ci,jj,v));
        s_Ci = size(Ci);
        s_Ci = prod(s_Ci(v));
        tmp(rr(jj)+1:rr(jj)+aug(2,jj),1:s_Ci) = mat_Ci;
        ss(jj) = ss(jj) + aug(2,jj);
        C1_hat = tensor(mat2tens(tmp,ss,jj),ss);
        
    end
    % set core tensor of U_tilde 
    if iscell(U_tilde{jj}) == 1
        U_tilde{jj}{end} = Y1{jj}{end};
    end
end
Y1{end} = C1_hat;
Y1{end-1} = eye(ss(end),ss(end));

U_tilde(1:m) = Y1(1:m);
U_tilde_SR(1:m) = Y1(1:m);

end

function [X] = func_ODE(C,F_tau,U0,t,A,d)
% function [X] = func_ODE(C,F_tau,U1,t,tau)
% This function defines the function F_tau(C(t)X U_0) X U0^*, for the
% tensor-ODE. Here C(t) is in tucker form, C0 = C X M_i, i.e. the M_i are
% matrices.

% argument of F_tau
m = length(U0);
s = size(C);
N = cell(1,m+2);
N{end} = C;
N{end-1} = eye(s(end),s(end));
N(1:m) = U0;

% apply F_tau
% F = F_tau(t,N,tau);
F = F_tau(t,N,A,d);

% multipl. with U0^*
dum = cell(1,m);
for i=1:m
    dum{i} = Mat0Mat0(U0{i},F{i});
end
X = ttm(F{end},dum,1:m);


end

function [C0_taui_hat] = C_zero_augment(Y1_i,Y0,i)

C0_taui_hat = Y0{i}{end};
tmp = sum(size(C0_taui_hat) == size(Y1_i{end}));
s = size(Y1_i{end});
if tmp < length(size(C0_taui_hat))
    if length(size(C0_taui_hat)) == 2
        C0_taui_hat(s(1),s(2)) = 0;
    elseif length(size(C0_taui_hat)) == 3
        C0_taui_hat(s(1),s(2),s(3)) = 0;
    elseif length(size(C0_taui_hat)) == 4
        C0_taui_hat(s(1),s(2),s(3),s(4)) = 0;
    elseif length(size(C0_taui_hat)) == 5
        C0_taui_hat(s(1),s(2),s(3),s(4),s(5)) = 0;
    end
end

end


function [C0_tau_hat] = C_zero_augment_core(C1,C0)

C0_tau_hat = C0;
tmp = sum(size(C0_tau_hat) == size(C1));
s = size(C1);
if tmp < length(size(C0_tau_hat))
    if length(size(C0_tau_hat)) == 2
        C0_tau_hat(s(1),s(2)) = 0;
    elseif length(size(C0_tau_hat)) == 3
        C0_tau_hat(s(1),s(2),s(3)) = 0;
    elseif length(size(C0_tau_hat)) == 4
        C0_tau_hat(s(1),s(2),s(3),s(4)) = 0;
    elseif length(size(C0_tau_hat)) == 5
        C0_tau_hat(s(1),s(2),s(3),s(4),s(5)) = 0;
    end
end

end
