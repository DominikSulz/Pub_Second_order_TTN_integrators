function [Y1,U_tilde,U_tilde_SR,C0_bar] = Second_order_parallel_for_TTN_Lanczos(tau,Y0,F_tau,t0,t1,A,d,r_min,root,U0_hat,Q_save,Q_hat_save,S_save,M_save)
% This function does one time-step with the second order parallel  
% integrator for TTNs in a recursuive way.

l_basis = 8;
Y1 = cell(size(Y0));
m = length(Y0) - 2;
U_tilde = cell(1,m+2);
U_tilde_SR = cell(1,m+2);
Ktau_i = cell(1,m);


%% Solving the ODEs - can be parallized 
for i=1:m
    if 1 == iscell(tau{i})
        Q0_i_hat = Q_hat_save{i}{end};
        Q0_i = Q_save{i}{end};
        S0_i_T = S_save{i}{end};
        Mi = M_save{i}{end};
    else
        Q0_i_hat = Q_hat_save{i};
        Q0_i = Q_save{i};
        S0_i_T = S_save{i};
        Mi = M_save{i};
    end
    
    if root
        F_tau_i = @(t,Y_tau_i,A,d) restriction(...
            F_tau(t,prolongation(Y_tau_i,U0_hat,i,Q0_i_hat),A,d),U0_hat,i,Q0_i_hat);
    else
        F_tau_i = @(t,Y_tau_i,A,d) restriction(...
            F_tau(t,prolongation(Y_tau_i,Y0,i,Q0_i_hat),A,d),Y0,i,Q0_i_hat);
    end

    % F_tau_i = @(t,Y_tau_i,A,d) restriction(...
    %     F_tau(t,prolongation(Y_tau_i,U0_hat,i,Q0_i_hat),A,d),U0_hat,i,Q0_i_hat);
    
    % project initial data on \hat U_0
    Y0_i = Ytau_i(tau{i},U0_hat{i},Mi*S0_i_T.'*Q0_i.'*Q0_i_hat); % initial value
    
    if 0 == iscell(tau{i})    % if \tau_i = l, l \in L 
         Y1_i = Lanczos_matrix(Y0_i,F_tau_i,t0,t1,A,d,l_basis);
         % Y1_i = RK_4_nonglobal(Y0_i,tau{i},F_tau_i,t0,t1,A,d);
    else % if \tau_i \notin L
        Y0_i{end-1} = eye(size(Q0_i_hat,2),size(Q0_i_hat,2));
        [Y1_i,U_tilde{i},U_tilde_SR{i},C0_bar_i] = Second_order_parallel_for_TTN(tau{i},Y0_i,F_tau_i,t0,t1,A,d,r_min,0,U0_hat{i},Q_save{i},Q_hat_save{i},S_save{i},M_save{i});
    end
    Ktau_i{i} = Y1_i;
    
    % distinguish between leaf and TTN case
    if 1 == iscell(Y1_i)
        % build up C0_tau_i_hat
        % C0_taui_hat =  C_zero_augment_core(Y1_i{end},C0_bar_i); % old

        % test
        m2 = length(Y1_i) - 2;
        E = cell(1,m2);
        for j = 1:m2
            E{j} = Mat0Mat0(Y1_i{j},U0_hat{i}{j});
        end
        C0_taui_hat = ttm(U0_hat{i}{end}, E, 1:m2);   % replaces zero padding
        
        % augment C_tau in 0-direction
        core1 = double(tenmat(C0_taui_hat,m2+1,1:m2)).'; 
        core2 = double(tenmat(Y1_i{end},m2+1,1:m2)).';
        rr = size(C0_taui_hat);
        
        % orthogonalization 
        W_taui_hat = qr_ordered(core1,core2); 
        [~,s] = size(core1);
        
        % set core tensor of U_tilde for recursion
        M = W_taui_hat(:,s+1:end);
        sz = size(Y1_i{end});
        [~,sz2] = size(W_taui_hat);
        sz(end) = sz2 - s;
        U_tilde{i}{end} = mat2tens(M.',sz,m2+1);
        U_tilde{i}{end} = tensor(U_tilde{i}{end},sz); 
        U_tilde{i}{end-1} = eye(sz(end),sz(end));
        
        U_tilde_SR{i}{end} = U_tilde{i}{end}; % new

        % retensorize
        [~,rr(end)] = size(W_taui_hat);
        Y1_i{end} = tensor(mat2tens(W_taui_hat.',rr,m2+1),rr);
        Y1{i} = Y1_i;
        Y1{end-1} = eye(rr(end),rr(end));

    else
        [~,rr] = size(U0_hat{i});
        Y1_i = qr_ordered(U0_hat{i},Y1_i); 

        Y1{i} = Y1_i;
        U_tilde{i} = Y1_i;
        U_tilde_SR{i} = Y1_i(:,rr+1:end); % new

    end
    
end

%% subflow \Psi

% solve the tensor ODE
if root
    C0_bar = U0_hat{end};
else
    C0_bar = Y0{end};
end
 
F_ODE = @(C0,F_tau,U0_hat_tau,t0,A,d) func_ODE(C0,F_tau,U0_hat(1:m),t0,A,d);

Y1{end-1} = eye(size(Y0{end-1}));
% C1_bar = RK_4_tensor_nonglobal(C0_bar,F_ODE,U0_hat(1:m),F_tau,t0,t1,tau,A,d);
C1_bar = Lanczos_tensor(C0_bar,F_ODE,U0_hat(1:m),F_tau,t0,t1,A,d,l_basis);

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

Ebar = cell(1,m); Etil = cell(1,m);
for j = 1:m
    if iscell(tau{j})
        Ebar{j} = Mat0Mat0(Y1{j},U0_hat{j});
        if aug(1,j)>0
            Etil{j} = Mat0Mat0(Y1{j},U_tilde_SR{j}); 
        end
    else     
        Ebar{j} = Y1{j}'*U0_hat{j};
        if aug(1,j)>0 
            Etil{j} = Y1{j}'*U_tilde_SR{j}; 
        end
    end
end

% augment \bar{C}_tau
C1_hat = ttm(C1_bar,Ebar,1:m); % first augment it to full size
szC = size(C1_bar);
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

        Ci_mat = Mat0Mat0(U_tilde_SR{jj},Ktau_i{jj})*Q0_i_hat.';
        Ci_size = size(C1_bar);  Ci_size(jj) = aug(2,jj);
        Ci  = tensor(mat2tens(Ci_mat,Ci_size,jj),Ci_size);

        Emb = Ebar;  Emb{jj} = Etil{jj};  
        C1_hat = C1_hat + ttm(Ci,Emb,1:m);
        
        % Ci_mat = Mat0Mat0(U_tilde_SR{jj},Ktau_i{jj})*Q0_i_hat.'; 
        % Ci_size = size(C1_bar);
        % Ci_size(jj) = aug(2,jj);
        % Ci = tensor(mat2tens(Ci_mat,Ci_size,jj),Ci_size);
        % 
        % Ctj = tensor(mat2tens(Ci,szC,j),szC);   % szC = size(C1_bar), szC(j)=ntil_j
        % Emb = Ebar; Emb{j} = Etil{j};
        % C1_hat = C1_hat + ttm(Ctj,Emb,1:m);

        % % augmentation old
        % tmp = double(tenmat(C1_hat,jj,v));
        % 
        % mat_Ci = double(tenmat(Ci,jj,v));
        % s_Ci = size(Ci);
        % s_Ci = prod(s_Ci(v));
        % tmp(rr(jj)+1:rr(jj)+aug(2,jj),1:s_Ci) = mat_Ci;
        % ss(jj) = ss(jj) + aug(2,jj);
        % C1_hat = tensor(mat2tens(tmp,ss,jj),ss);
        
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


%% old
%%%%%% test
% M_save = cell(1,m);
% for jj=1:m
%     M_save{jj} = Mat0Mat0(U0_hat{jj},Y0{jj});
% end
% C0_bar = ttm(Y0{end},M_save,1:m);

% C0_bar = C_zero_augment_core(U0_hat{end},Y0{end});
%%%%%%%