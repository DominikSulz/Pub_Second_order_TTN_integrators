function[X_new,X_new_ad] = ref_sol_spin_systems_aug_BUG(X0,tau,dt,t0,T,A,F,d,tol,r_min,r_max)

t_steps = (T-t0)/dt;
X_start = X0;
X_start_ad = X0;

for it=1:t_steps
    t0 = (it-1)*dt;
    t1 = it*dt;

    % rank-adaptive BUG
    % X_new = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X_start,F,t0,t1,A,d,r_min);
    % X_new = truncate(X_new,tol,r_max,r_min);
    
    % 2nd order parallel BUG
    [U0_hat,M_save,CF_save] = pre_augment(tau,X_start,X_start,F,t0,t1,A,d,r_min,1);
    [Q_save,Q_hat_save,S_save] = pre_construct_Q(tau,X_start,U0_hat,M_save,CF_save,F,t0,t1,A,d,r_min,1);
    X_new = Second_order_parallel_for_TTN(tau,X_start,F,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save,M_save);
    X_new = truncate(X_new,tol,r_max,r_min);

    % 2nd order augmented BUG
    [U0_hat,M_save,CF_save] = pre_augment(tau,X_start_ad,X_start_ad,F,t0,t1,A,d,r_min,1);
    [Q_save,Q_hat_save,S_save] = pre_construct_Q(tau,X_start_ad,U0_hat,M_save,CF_save,F,t0,t1,A,d,r_min,1);
    X_new_ad = Second_order_rank_adaptive_for_TTN(tau,X_start_ad,F,t0,t1,A,d,r_min,1,U0_hat,Q_save,Q_hat_save,S_save,M_save);
    X_new_ad = truncate(X_new_ad,tol,r_max,r_min);

    % set for new time step
    X_start = X_new;
    X_start_ad = X_new_ad;
end

end