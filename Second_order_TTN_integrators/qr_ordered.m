function [U_hat] = qr_ordered(U0,U1)

% assure that U0 is orthogonal
[Q,R] = qr(U0,0);
Q = Q*diag(sign(diag(R)) + (diag(R)==0));   % Eichung/Vorzeichen erhalten
U0 = Q; 

% Project U1 on the nullspace of U0 
U_tmp = U1 - U0*(U0'*U1);
U_tmp = U_tmp - U0*(U0'*U_tmp); % second iteration of projection 

% orthonormalize U_tmp
[U1_orth,S,~] = svd(U_tmp,0);
tol = 10^-10 * norm(U1);
rank_new = sum(diag(S) > tol);
U1_orth = U1_orth(:,1:rank_new);

% augment
U_hat = [U0 U1_orth];

% % --- Diagnose ---
% d_in  = norm(U0'*U0 - eye(size(U0,2)),'fro');          % Input orthonormal?
% d_out = norm(U0'*U1_orth,'fro');                        % Blockorthogonalitaet
% fprintf('qr_ordered: dim=%dx%d -> +%d | d_in=%.2e d_out=%.2e | s1=%.3e s_last=%.3e\n', ...
%     size(U0,1), size(U0,2), rank_new, d_in, d_out, S(1,1), ...
%     max(S(min(rank_new,end),min(rank_new,end)),0));

end