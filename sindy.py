from numpy.core.numeric import argwhere
from bicycle_robot import BicycleRobot
import numpy as np

from diff_drive_robot import DifferentialDriveRobot
from diff_drive_robot_2 import DifferentialDriveRobot2

from sindy.constant_term import SindyBasisConstantTermGenerator
from sindy.linear_terms import SindyBasisLinearTermsGenerator
from sindy.quadratic_terms import SindyBasisQuadraticTermsGenerator
from sindy.sin_terms import SindyBasisSinTermsGenerator
from sindy.cos_terms import SindyBasisCosTermsGenerator
from sindy.tan_terms import SindyBasisTanTermsGenerator

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
    n = s_data.shape[1]
    m = u_data.shape[1]

    basis_gens = [
        SindyBasisConstantTermGenerator(),
        SindyBasisLinearTermsGenerator(n,m),
        SindyBasisQuadraticTermsGenerator(n,m),
        SindyBasisSinTermsGenerator(n,m),
        SindyBasisCosTermsGenerator(n,m),
        SindyBasisTanTermsGenerator(n,m)
    ]

    B_cols = sum(map(lambda gen: gen.numTerms(), basis_gens))

    B = np.empty((N, B_cols), dtype=np.float64)
    col = 0
    for gen in basis_gens:
        B, col = gen.addToBasis(s_data, u_data, B, col)
    #/

    assert col == B.shape[1]    
    return B, basis_gens
# /generateSindyBasisFunctions()

def main():
    # robot = DifferentialDriveRobot(radius=15,
    #                                wheel_radius=6,
    #                                wheel_thickness=3)
    # robot.reset(40, 40, 0)

    robot = DifferentialDriveRobot2(radius=15,
                                    wheel_radius=6,
                                    wheel_thickness=3)
    robot.reset(40, 40, 0, 0, 0)

    # robot = BicycleRobot(wheel_radius=20, baseline=60)
    # robot.reset(40, 40, 0)

    n_trials = 10000
    n_samples_per_u = 20
    dt = 0.001
    t_data, s_data, u_data, s_dot_data = \
        generateSindyData(robot, n_trials, dt, n_samples_per_u)
    B, basis_gens = generateSindyBasisFunctions(t_data, s_data, u_data)
    indices = np.arange(B.shape[1])
    threshold = 1.0e-2
    print('Iter -1 n_basis = {}'.format(B.shape[1]))
    for iter in range(10):
        coeffs, residuals, rank, _ = np.linalg.lstsq(B, s_dot_data)
        argwhere_above_threshold = []
        for i in range(coeffs.shape[0]):
            if np.max(np.abs(coeffs[i,:])) > threshold:
                argwhere_above_threshold.append(i)
            # /if
        # /for
        print('Iter {} rank = {} max_resid = {} n_basis = {}'
              .format(iter, rank, np.max(residuals) if len(residuals) > 0 else np.nan, len(argwhere_above_threshold)))
        # if len(argwhere_above_threshold) == len(coeffs):
        #     break
        B = B[:, argwhere_above_threshold]
        indices = indices[argwhere_above_threshold]
    # /for iter
    coeffs = coeffs[argwhere_above_threshold, :]
    
    print('Final norm = ', np.linalg.norm(s_dot_data - B@coeffs))
    print('coeffs |min|, |max| =', np.min(np.abs(coeffs)), np.max(np.abs(coeffs)))
    print('indices =', indices)
    assert len(indices) > 0

    n_state = s_data.shape[1]
    n_control = u_data.shape[1]

    terms = []
    indices_idx = 0
    col = 0

    for gen in basis_gens:
        gen_terms, indices_idx, col = \
            gen.extractTerms(indices, indices_idx, col, robot.stateNames(), robot.controlNames())
        terms += gen_terms
    # /for gen
    print('fn =', terms)

    for i in range(n_state):
        which_terms = np.nonzero(coeffs[:,i] > threshold)[0]
        expr = ''
        for k in which_terms:
            expr += '{:.2f}*{} + '.format(coeffs[k,i], terms[k])
        #/
        print(f'{robot.stateNames()[i]}_dot =', expr[:-3])
    # /for i

# /main()

if __name__ == "__main__":
    main()
