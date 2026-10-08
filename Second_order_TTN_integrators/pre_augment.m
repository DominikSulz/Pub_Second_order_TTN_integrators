function [U0_hat,M_save,CF_save] = pre_augment(tau,Y0,Y0_save,F_tau,t0,t1,A,d,r_min,root)

m = length(Y0) - 2;
U0_hat = cell(1,m+2);
M_save = cell(1,m+1);
CF_save = cell(1,m+1);

%% Pre-augmentation
F_eval = F_tau(t0,Y0,A,d);

for ii=1:m
    % subflow \Phi_i
    v = 1:m+1;
    v = v(v~=ii);
    Mat_C = tenmat(Y0{end},ii,v);
    [Q0_i,Si] = qr(double(Mat_C).',0);
    
    % U0_i = Y0{ii}; % orthogonal basis before moving of orthogonal center
    % Y0{ii} = Ytau_i(tau{ii},Y0{ii},Si.'); % U0*S0
    Y0_i = Ytau_i(tau{ii},Y0{ii},Si.'); % U0*S0

    F_tau_i = @(t,Y_tau_i,A,d) restriction(...
        F_tau(t,prolongation(Y_tau_i,Y0,ii,Q0_i),A,d),Y0,ii,Q0_i);
    
    if iscell(Y0{ii}) == 0 
        U0_hat{ii} = qr_ordered(Y0_save{ii},F_tau_i(t0,Y0_i,A,d));
    else
        % recursively building the augmentation
        [U0_hat{ii},M_save{ii},CF_save{ii}] = pre_augment(tau{ii},Y0_i,Y0_save{ii},F_tau_i,t0,t1,A,d,r_min,0);
    end

    F_eval{ii} = Mat0Mat0(U0_hat{ii},F_eval{ii});
    if iscell(Y0{ii}) == 0
        M_save{ii} = Mat0Mat0(U0_hat{ii},Y0_save{ii});
    else
        M_save{ii}{end} = Mat0Mat0(U0_hat{ii},Y0_save{ii});
    end
end

%% Build C0_tau_hat^0 and \bar C^0
tmpM = cell(1,m);
for ii=1:m
    if iscell(Y0{ii}) == 0
        tmpM{ii} = M_save{ii};
    else
        tmpM{ii} = M_save{ii}{end};
    end
end

% tmp1 = ttm(Y0{end},tmpM,1:m);
tmp1 = ttm(Y0_save{end},tmpM,1:m);

% Build C0_tau_hat^0
tmp1_mat = double(tenmat(tmp1,m+1,1:m));
CF_hat = ttm(F_eval{end},F_eval(1:m),1:m);
CF_save{end} = CF_hat;
if ~root
    tmp2 = double(tenmat(CF_hat,m+1,1:m));
    C0_tau_hat = qr_ordered(tmp1_mat.',tmp2.');
    sz = size(CF_hat);
    [~,sz(end)] = size(C0_tau_hat);
    U0_hat{end-1} = eye(sz(end),sz(end));
    U0_hat{end} = mat2tens(C0_tau_hat.',sz,m+1,1:m);
else 
    U0_hat{end-1} = 1;
    U0_hat{end} = tmp1;
end



end




%%% old code for Q, S and \hat Q

% %% Man muss vielleicht Q_hat_i nochmal von neuem konstruieren
% 
% 
% %% Build Q_hat_i
% for ii=1:m
%     v = 1:m+1;
%     v = v(v~=ii);
%     vv = 1:m;
%     vv = vv(vv~=ii);
% 
%     C0_Mx_i = ttm(Y0{end},tmpM(vv),vv);
%     Mat_C = tenmat(C0_Mx_i,ii,v);
%     [Q0_i,S0_i_T] = qr(double(Mat_C).',0);
% 
%     if iscell(Y0{ii}) == 0
%         C_UU = ttm(CF_hat,M_save{ii}',ii);
%     else
%         C_UU = ttm(CF_hat,M_save{ii}{end}',ii);
%     end
%     % [Q0_i_hat,~] = qr([Q0_i,double(tenmat(C_UU,i,v)).'],0);
%     Q0_i_hat = qr_ordered(Q0_i,double(tenmat(C_UU,ii,v)).');
% 
%     % norm((eye(size(Q0_i,1)) - Q0_i*Q0_i') ...
%     %  * double(tenmat(C_UU,ii,v)).','fro')
% 
%     if iscell(Y0{ii}) == 0
%         Q_save{ii} = Q0_i;
%         Q_hat_save{ii} = Q0_i_hat;
%         S_save{ii} = S0_i_T;
%     else
%         Q_save{ii}{end} = Q0_i;
%         Q_hat_save{ii}{end} = Q0_i_hat;
%         S_save{ii}{end} = S0_i_T;
%     end
% 
%     % % Build initial data
%     % Y0_i = Ytau_i(tau{iiu},Y0{ii},S0_i_T.'); % initial value for K-step 
%     % 
%     % if 0 == iscell(tau{ii}) 
%     %     init{ii} = U0_hat{ii}*Mat0Mat0(U0_hat{ii},Y0_i)*Q0_i.'*Q0_i_hat;
%     % else
%     %     tmp = Mat0Mat0(U0_hat{ii},Y0_i)*Q0_i.'*Q0_i_hat;
%     %     m1 = length(size(U0_hat{ii}{end}));
%     %     init{ii}{end} = ttm(U0_hat{ii}{end},tmp,m1);
%     % end
% end