function [Q_save,Q_hat_save,S_save] = pre_construct_Q(tau,Y0,U0_hat,M_save,CF_save,F_tau,t0,t1,A,d,r_min,root)
                                                    
m = length(Y0) - 2;
Q_save = cell(1,m+1);
Q_hat_save = cell(1,m+1);
S_save = cell(1,m+1);
tmpM = cell(1,m);

% 15. Juli added
% tmp = eye(size(U0_hat{end},m+1),size(Y0{end},m+1)); % test 8.7.26


% if ~root
%     Proj0 = Mat0Mat0(U0_hat,Y0); % equivalent to line above if \hatU^*U = I
% else 
%     Proj0 = 1;
% end
% CF_hat = CF_save{end};


% CF_hat = ttm(CF_hat,Proj0,m+1);

% Construct CF_hat in \hat Q Basis
F_eval = F_tau(t0,Y0,A,d);
for ii=1:m
    F_eval{ii} = Mat0Mat0(U0_hat{ii},F_eval{ii});
end
CF_hat = ttm(F_eval{end},F_eval(1:m),1:m);

for ii=1:m
    if iscell(Y0{ii}) == 0
        tmpM{ii} = M_save{ii};
    else
        tmpM{ii} = M_save{ii}{end};
    end
end
% tmpM{end+1} = Proj0;

for ii=1:m
    v = 1:m+1;
    v = v(v~=ii);
    vv = 1:m;
    vv = vv(vv~=ii);

    % Mat_C = tenmat(Y0{end},ii,v);
    % [Q0_C0,S0_C0] = qr(double(Mat_C).',0);
    
    % C0_Mx_i = ttm(Y0{end},tmpM(v),v); 
    % Y0_aug0 = F_Id(t0,Y0,A,d); % test
    C0_Mx_i = ttm(Y0{end},tmpM(vv),vv); 
    Mat_C = tenmat(C0_Mx_i,ii,v);
    [Q0_i,S0_i_T] = qr(double(Mat_C).',0);

    % % test 8.7.26
    % U0U0_hat = Mat0Mat0(Y0{ii},U0_hat{ii});
    % C0_Mx_i = ttm(U0_hat{end},U0U0_hat,ii);
    % Mat_C = tenmat(C0_Mx_i,ii,v);
    % [Q0_i,S0_i_T] = qr(double(Mat_C).',0);

    if iscell(Y0{ii}) == 0
        C_UU = ttm(CF_hat,M_save{ii}',ii);
    else
        C_UU = ttm(CF_hat,M_save{ii}{end}',ii);
    end
    Q0_i_hat = qr_ordered(Q0_i,double(tenmat(C_UU,ii,v)).');

    if iscell(Y0{ii}) == 0
        Q_save{ii} = Q0_i;
        Q_hat_save{ii} = Q0_i_hat;
        S_save{ii} = S0_i_T;
    else

        Y0_i= Ytau_i(tau{ii},Y0{ii},(S0_i_T.'*Q0_i.'*conj(Q0_i_hat))); % hier evtl. conj(Q0_i_hat)?
        
        Env = U0_hat;
        sz  = size(U0_hat{end});
        sz(end) = size(Y0{end},m+1);        % hat_r_tau: Q-hat-Seite der Elternkante
        Env{end}   = tenzeros(sz);          % nur die Groesse wird gelesen
        Env{end-1} = eye(sz(end));

        F_tau_i = @(t,Y_tau_i,A,d) restriction(...
            F_tau(t,prolongation(Y_tau_i,Env,ii,Q0_i_hat),A,d),Env,ii,Q0_i_hat);

        % F_Id_i = @(t,Y_tau_i,A,d) restriction(...
        %     F_Id(t,prolongation(Y_tau_i,Y0,ii,Q0_C0),A,d),U0_hat,ii,Q0_C0);

        % F_tau_i = @(t,Y_tau_i,A,d) restriction(...
        %     F_tau(t,prolongation(Y_tau_i,Y0,ii,Q0_C0),A,d),U0_hat,ii,Q0_i_hat);

        [Q_save{ii},Q_hat_save{ii},S_save{ii}] = pre_construct_Q(tau{ii},Y0_i,U0_hat{ii},M_save{ii},CF_save{ii},F_tau_i,t0,t1,A,d,r_min,0);
        Q_save{ii}{end} = Q0_i;
        Q_hat_save{ii}{end} = Q0_i_hat;
        S_save{ii}{end} = S0_i_T;
    end

end

end


% function [C0_tau_hat] = C_zero_augment_core(C1,C0)
% 
% C0_tau_hat = C0;
% tmp = sum(size(C0_tau_hat) == size(C1));
% s = size(C1);
% if tmp < length(size(C0_tau_hat))
%     if length(size(C0_tau_hat)) == 2
%         C0_tau_hat(s(1),s(2)) = 0;
%     elseif length(size(C0_tau_hat)) == 3
%         C0_tau_hat(s(1),s(2),s(3)) = 0;
%     elseif length(size(C0_tau_hat)) == 4
%         C0_tau_hat(s(1),s(2),s(3),s(4)) = 0;
%     elseif length(size(C0_tau_hat)) == 5
%         C0_tau_hat(s(1),s(2),s(3),s(4),s(5)) = 0;
%     end
% end
% 
% end



% old 
% CF_hat = cell(1,m+1);
% 
% F_eval = F_tau(t0,Y0,A,d);
% % if root
% %     F_eval = rounding(F_eval,tau);
% % end
% 
% % construct CF_hat and tmpM
% for ii=1:m
%     F_eval{ii} = Mat0Mat0(U0_hat{ii},F_eval{ii}); % F in augmented basis
% 
%     if iscell(Y0{ii}) == 0
%         tmpM{ii} = M_save{ii};
%     else
%         tmpM{ii} = M_save{ii}{end};
%     end
% end
% CF_hat = ttm(F_eval{end},F_eval(1:m),1:m);

% % augment CF_hat in 0-dim with zeros



% old 
% % check if Q0_i_hat violates the rank condition
% rr = size(C_UU);
% max_r = prod(rr)/prod(rr(v));
% [~,ss] = size(Q0_i_hat);
% if ss > max_r
%     Q0_i_hat = Q0_i_hat(:,max_r);
% end
