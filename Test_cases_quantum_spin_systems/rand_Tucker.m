function [Y,tau] = rand_Tucker(n_l,r_l,r_tau,d)
% This function creates a random TTN with wanted sizes

%% Y
% generate the TTN
Y = cell(1,d+2);
tau = cell(1,d);
r_full = [r_tau,1];
C0 = tensor(rand(r_full),r_full) + 1i*tensor(rand(r_full),r_full);
Mat = tenmat(C0,d+1,1:d);
[Mat,~] = qr(double(Mat).',0);
C0 = tensor(mat2tens(Mat.',r_tau,d+1,1:d),[r_tau 1]);
Y{end-1} = 1; %identity
Y{end} = C0;

for ii=1:d
    Y{ii} = orth(rand(n_l(ii),r_l(ii)) + 1i*rand(n_l(ii),r_l(ii)));
end

end

