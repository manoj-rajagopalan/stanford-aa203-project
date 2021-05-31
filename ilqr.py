from jax.core import as_named_shape
import numpy as np
import jax
import jax.numpy as jnp
import jaxlib

n_state = 3
n_control = 2

def iLQR_policy(mat_A, mat_B,
                mat_Css, mat_Cus, mat_Cuu, vec_Cs, vec_Cu,
                mat_Jss, vec_Js):
    '''
    '''
    R_BtJB_inv = np.linalg.inv(mat_Cuu + mat_B.T @ mat_Jss @ mat_B)
    mat_state_coeff = mat_Cus + 2 * mat_B.T @ mat_Jss @ mat_A
    vec_const = vec_Cu + mat_B.T @ vec_Js
    mat_L = -0.5 * R_BtJB_inv @ mat_state_coeff
    vec_l = -0.5 * R_BtJB_inv @ vec_const
    return mat_L, vec_l
# /iLQR_iteration

def iLQR_value(s,
               mat_L, vec_l,
               mat_A, mat_B,
               mat_Css, mat_Cus, mat_Cuu,
               vec_Cs, vec_Cu, scalar_C0,
               mat_Jss, vec_Js, scalar_J0):
    '''
        Value per-stage is of the form J_k(s) = s^T @ mat_Jss @ s + dot(vec_Js, s) + scalar_J0
    '''
    mat_A_plus_BL = mat_A + mat_B @ mat_L

    mat_Jss = mat_Css \
            + mat_L.T @ mat_Cus \
            + mat_L.T @ mat_Cuu @ mat_L \
            + mat_A_plus_BL.T @ mat_Jss @ mat_A_plus_BL

    vec_Bl = mat_B @ vec_l # intermediate

    vec_Js = mat_Cus.T @ vec_l \
           + 2 * vec_l @ mat_Cuu @ mat_L \
           + vec_Cs \
           + mat_L.T @ vec_Cu \
           + 2 * vec_Bl @ mat_Jss @ mat_A_plus_BL \
           + mat_A_plus_BL.T @ vec_Js

    scalar_J0 = vec_l @ mat_Cuu @ vec_l \
              + np.dot(vec_Cu, vec_l) \
              + scalar_C0 \
              + vec_Bl @ mat_Jss @ vec_Bl \
              + np.dot(vec_Js, vec_Bl) \
              + scalar_J0

    assert len(mat_Jss.shape) == 2
    assert len(vec_Js.shape) == 1
    return mat_Jss, vec_Js, scalar_J0
# /iLQR_cost()

def stageCostComponents(Q, R, s_bar, u_bar, s_goal):
    '''
    '''
    delta_sbar = s_bar - s_goal
    mat_Css = 0.5 * Q
    mat_Cus = np.zeros((n_control, n_state))
    mat_Cuu = 0.5 * R
    vec_Cs = Q @ delta_sbar
    vec_Cu = R @ u_bar
    scalar_C0 = 0.5 * np.dot(vec_Cs, delta_sbar) + np.dot(vec_Cu, u_bar)
    return mat_Css, mat_Cus, mat_Cuu, vec_Cs, vec_Cu, scalar_C0
# /stageCostComponents()

def totalCost(s, u, s_goal, N, P_N, Q_k, R_k):
    J = 0
    for k in range(N):
        J += (s[k] - s_goal) @ Q_k @ (s[k] - s_goal) + u[k] @ R_k @ u[k]
    # /for k
    J += (s[N] - s_goal) @ P_N @ (s[N] - s_goal)
    return 0.5 * J
# /totalCost()

def iLQR(f, df_ds, df_du, s0, s_goal, N, P_N, Q_k, R_k):
    '''
    f: (nonlinear) dynamics function
    df_ds, df_du: Jacobian of f w.r.t. state and control
    s0, s_goal: initial and goal states
    N : number of time-steps per episode
    P_N : terminal cost coefficient
    Q_k : state-cost coefficient per-stage
    R_k : control-cost coefficient per-stage
    '''
    num_episodes = 100
    u_convergence_tol = 1.0e-4

    # Initialize trajectory: nominal and perturbed
    s_bar = np.zeros((N+1, n_state))
    u_bar = np.zeros((N, n_control))
    # s_bar, u_bar = getInitialTrajectory(mat_A, mat_B, s0, s_goal)
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = f(s_bar[k], u_bar[k])
    # /for k
    s = s_bar.copy() # initial perturbations are zero
    u = u_bar.copy()

    mat_Ls = np.zeros((N, n_control, n_state))
    vec_ls = np.zeros((N, n_control))

    for episode in range(num_episodes):

        # Express terminal cost in standard per-stage structure
        delta_sbar_N = s[N] - s_bar[N]
        mat_Jss = 0.5 * P_N
        vec_Js = P_N @ delta_sbar_N
        scalar_J0 = 0.5 * np.dot(delta_sbar_N, vec_Js)

        # Riccati recursion
        for k in range(N-1,-1,-1):
            mat_Css, mat_Cus, mat_Cuu, vec_Cs, vec_Cu, scalar_C0 = \
                stageCostComponents(Q_k, R_k, s_bar[k], u_bar[k], s_goal)
            mat_A = df_ds(s_bar[k], u_bar[k])
            mat_B = df_du(s_bar[k], u_bar[k])
            mat_Ls[k], vec_ls[k] = \
                iLQR_policy(mat_A, mat_B, mat_Css, mat_Cus, mat_Cuu, vec_Cs, vec_Cu, mat_Jss, vec_Js)
            mat_Jss, vec_Js, scalar_J0 = \
                iLQR_value(s,
                        mat_Ls[k], vec_ls[k],
                        mat_A, mat_B,
                        mat_Css, mat_Cus, mat_Cuu, vec_Cs, vec_Cu, scalar_C0,
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
        J = totalCost(s, u, s_goal, N, P_N, Q_k, R_k)
        u_diff_norm = np.linalg.norm(u - u_bar)
        print('Episode {} cost = {} |s_N - s*| = {} max |delta_u| = {}'
              .format(episode, J, np.linalg.norm(s[N]-s_goal), u_diff_norm))

        if u_diff_norm < u_convergence_tol:
            break
        else:
            s_bar = s.copy()
            u_bar = u.copy()
    # /for episode

    return s, u
# /iLQR()
