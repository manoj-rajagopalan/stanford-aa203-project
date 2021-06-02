import numpy as np

from diff_drive_robot import DifferentialDriveRobot

def generateSindyData(robot, n_trials, dt, n_samples_per_u=5):
    n_state = robot.stateDim()
    n_control = robot.controlDim()
    t_data = dt * np.cumsum(1 + np.arange(n_samples_per_u * n_trials))
    s_data = np.empty((n_samples_per_u * n_trials, n_state))
    s_dot_data = np.empty_like(s_data)
    u_data = np.empty((n_samples_per_u * n_trials, n_control))
    u_min, u_max = robot.controlLimits()
    u_range = u_max - u_min
    temp_s_data = np.empty((n_samples_per_u+2, n_state))
    temp_s_data[-1] = robot.s[-1]
    k = 0 # index into data tables
    for n in range(n_trials):
        temp_s_data[0] = temp_s_data[-1]
        u = u_min + np.random.random(2) * u_range
        for nu in range(n_samples_per_u + 1): # samples per fixed control
            temp_s_data[nu+1] = \
                robot.applyControl(dt, temp_s_data[nu], u)
        # /for nu
        s_data[k:k+n_samples_per_u] = temp_s_data[1:-1]
        u_data[k:k+n_samples_per_u, :] = u[np.newaxis,:] # broadcast
        s_dot_data[k:k+n_samples_per_u] = (temp_s_data[2:] - temp_s_data[:-2]) / (2 * dt)
        k += n_samples_per_u
    # /for n

    return t_data, s_data, u_data, s_dot_data

# /generateSindyData()

def generateSindyBasisFunctions(t_data, s_data, u_data):
    N = len(t_data)
    assert N == len(s_data)
    assert N == len(u_data)

    su_data = np.concatenate((s_data, u_data), axis=1)
    B = np.ones((N,1), dtype='float64')
    B = np.concatenate((B, su_data), axis=1)
    indices = list(range(su_data.shape[1]))
    sin_su_data = np.sin(su_data)
    cos_su_data = np.cos(su_data)
    B = np.concatenate((B, sin_su_data), axis=1)
    B = np.concatenate((B, cos_su_data), axis=1)
    for i in range(len(indices)):
        B = np.concatenate((B, su_data * su_data[:, indices]), axis=1)
        B = np.concatenate((B, su_data * sin_su_data[:, indices]), axis=1)
        B = np.concatenate((B, su_data * cos_su_data[:, indices]), axis=1)
        indices = np.roll(indices, -1)
    # /for i
    return B
# /generateSindyBasisFunctions()

def main():
    robot = DifferentialDriveRobot(radius=15,
                                   wheel_radius=6,
                                   wheel_thickness=3)
    robot.reset(20, 40, 0)
    n_trials = 1000
    n_samples_per_u = 5
    dt = 0.01
    t_data, s_data, u_data, s_dot_data = \
        generateSindyData(robot, n_trials, dt, n_samples_per_u)
    B = generateSindyBasisFunctions(t_data, s_data, u_data)
    threshold = 1.0e-3
    print('Iter -1 n_basis = {}'.format(B.shape[1]))
    for iter in range(100):
        coeffs, residuals, rank, _ = np.linalg.lstsq(B, s_dot_data)
        argwhere_above_threshold = []
        for i, coeff in enumerate(coeffs):
            if np.linalg.norm(coeff, ord=np.inf) > threshold:
                argwhere_above_threshold.append(i)
            # /if
        # /for
        print('Iter {} rank = {} max_resid = {} n_basis = {}'
              .format(iter, rank, np.max(residuals) if len(residuals) > 0 else np.nan, len(argwhere_above_threshold)))
        if len(argwhere_above_threshold) == len(coeffs):
            break
        B = B[:, argwhere_above_threshold]
    # /for iter
    
    print('Final norm = ', np.linalg.norm(s_dot_data - B@coeffs))
# /main()

if __name__ == "__main__":
    main()
