import numpy as np

def iLQR_iteration(mat_A, mat_B, 
                   mat_Css, mat_Cus, mat_Csu, mat_Cuu, vec_Cs, vec_Cu, scalar_C0,
                   mat_Jss, vec_Js, scalar_J0):
    '''
    '''

    BtJ = mat_B.T @ mat_Jss
    G_inv = np.linalg.inv(2 * (mat_Cuu + BtJ @ mat_B))
    H = mat_Cus + mat_Csu.T + 2 * BtJ @ mat_A
    r = vec_Cu + mat_B.T @ vec_Js

    mat_L = -G_inv @ H
    vec_l = -G_inv @ r

    mat_F = mat_A + mat_B @ mat_L
    vec_p = mat_B @ vec_l

    mat_Jss = mat_Css \
            + mat_L.T @ mat_Cus \
            + mat_Csu @ mat_L \
            + mat_L.T @ mat_Cuu @ mat_L \
            + mat_F.T @ mat_Jss @ mat_F

    vec_Js = (mat_Cus.T + mat_Csu) @ vec_l \
           + 2 * mat_L.T @ mat_Cuu @ vec_l \
           + vec_Cs \
           + mat_L.T @ vec_Cu \
           + 2 * mat_F.T @ mat_Jss @ vec_p \
           + mat_F.T @ vec_Js

    scalar_J0 = np.dot(mat_Cuu @ vec_l + vec_Cu, vec_l) \
              + scalar_C0 \
              + np.dot(mat_Jss @ vec_p + vec_Js, vec_p) \
              + scalar_J0

    assert len(mat_Jss.shape) == 2
    assert len(vec_Js.shape) == 1
    return mat_L, vec_l, mat_Jss, vec_Js, scalar_J0
# /iLQR_iteration


def stageCostComponents(Q, R, R_delta_u, s_bar, u_bar, s_goal):
    '''
    '''
    n_state = len(s_bar)
    n_control = len(u_bar)
    delta_sbar = s_bar - s_goal
    mat_Css = 0.5 * Q
    mat_Cus = np.zeros((n_control, n_state), dtype='float64')
    mat_Csu = mat_Cus.T
    mat_Cuu = 0.5 * (R + R_delta_u)
    vec_Cs = Q @ delta_sbar
    vec_Cu = R @ u_bar
    scalar_C0 = 0.5 * np.dot(vec_Cs, delta_sbar) + np.dot(vec_Cu, u_bar)
    return mat_Css, mat_Cus, mat_Csu, mat_Cuu, vec_Cs, vec_Cu, scalar_C0
# /stageCostComponents()


def totalCost(s, u, s_goal, u_bar, N, P_N, Q, R_k, R_delta_u):
    J = 0
    for k in range(N):
        delta_u = u[k] - u_bar[k]
        J += (s[k] - s_goal) @ Q[k] @ (s[k] - s_goal) \
           + u[k] @ R_k @ u[k] \
           + delta_u @ R_delta_u @ delta_u
    # /for k
    J += (s[N] - s_goal) @ P_N @ (s[N] - s_goal)
    return 0.5 * J
# /totalCost()

def iLQR(model, s0, s_goal, N, dt, P_N, Q, R_k, R_delta_u, n_iter):
    '''
    model: dynamics and jacobians of dynamics
    s0, s_goal: initial and goal states
    N : number of time-steps per episode
    P_N : terminal cost coefficient
    Q_k : state-cost coefficient per-stage
    R_k : control-cost coefficient per-stage
    '''
    t_dummy = np.nan # not used

    # State-transition function and its Jacobians, from dynamics
    f = lambda s,u: s + dt * model.dynamics(t_dummy, s, u)
    # Linearize model dynamics ODE into A*dx + B*du
    A = model.dynamicsJacobianWrtState
    B = model.dynamicsJacobianWrtControl
    # ... and compute approximate Jacobians for transition function
    df_ds = lambda s,u: np.eye(model.stateDim()) + dt * A(s,u)
    df_du = lambda s,u: dt * B(s,u)

    u_convergence_tol = 1.0e-1
    n_state = len(s_goal)
    n_control = R_k.shape[-1]

    # Initialize trajectory: nominal and perturbed
    s_bar = np.zeros((N+1, n_state), dtype='float64')
    u_bar = np.zeros((N, n_control), dtype='float64')
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = f(s_bar[k], u_bar[k])
    # /for k
    s = s_bar.copy() # initial perturbations are zero
    u = u_bar.copy()

    mat_Ls = np.zeros((N, n_control, n_state), dtype='float64')
    vec_ls = np.zeros((N, n_control), dtype='float64')

    cost_history = np.zeros(n_iter)
    sf_norm_history = np.zeros_like(cost_history)
    du_norm_history = np.zeros_like(cost_history)

    for iter in range(n_iter):

        # Express terminal cost in standard per-stage structure
        delta_sbar_N = s_bar[N] - s_goal
        mat_Jss = 0.5 * P_N
        vec_Js = P_N @ delta_sbar_N
        scalar_J0 = 0.5 * np.dot(delta_sbar_N, vec_Js)

        # Riccati recursion
        for k in range(N-1,-1,-1):
            mat_Css, mat_Cus, mat_Csu, mat_Cuu, vec_Cs, vec_Cu, scalar_C0 = \
                stageCostComponents(Q[k], R_k, R_delta_u, s_bar[k], u_bar[k], s_goal)
            mat_A = df_ds(s_bar[k], u_bar[k])
            mat_B = df_du(s_bar[k], u_bar[k])
            mat_Ls[k], vec_ls[k], mat_Jss, vec_Js, scalar_J0 = \
                iLQR_iteration(mat_A, mat_B,
                               mat_Css, mat_Cus, mat_Csu, mat_Cuu, vec_Cs, vec_Cu, scalar_C0,
                               mat_Jss, vec_Js, scalar_J0)
        # /for k

        # Forward-integrate dynamics with new controls
        assert (s_bar[0] == s0).all()
        for k in range(N):
            delta_s = s[k] - s_bar[k]
            delta_u = mat_Ls[k] @ delta_s + vec_ls[k]
            u[k] = u_bar[k] + delta_u
            s[k+1] = f(s[k], u[k])
        # /for k
        cost_history[iter] = totalCost(s, u, s_goal, u_bar, N, P_N, Q, R_k, R_delta_u)
        sf_norm_history[iter] = np.linalg.norm(s[N]-s_goal)
        du_norm_history[iter] = np.max(np.abs(u - u_bar))

        print('Episode {} cost = {} ||s_N - s*||_2 = {} ||delta_u||_inf = {}'
              .format(iter, cost_history[iter], sf_norm_history[iter], du_norm_history[iter]))

        if du_norm_history[iter] < u_convergence_tol:
            break
        else:
            s_bar = s.copy()
            u_bar = u.copy()
    # /for episode

    metrics_history = {'cost': cost_history,
                       'sf_norm': sf_norm_history,
                       'du_norm': du_norm_history}
    return s, u, mat_Ls, vec_ls, metrics_history
# /iLQR()
