function [X,tau] = init_spin_up_and_down(r,cc,d)

[X1,tau] = init_spin_all_dim_same_rank(r,cc,d);

X2 = init_spin_all_dim_same_rank_down(r,cc,d);
X2{end} = 0.01*X2{end};

X = Add_TTN(X1,X2,tau);
X = rounding(X,tau);

end