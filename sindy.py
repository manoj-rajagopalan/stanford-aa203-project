import numpy as np

from sindy.constant_term import SindyBasisConstantTermGenerator
from sindy.linear_terms import SindyBasisLinearTermsGenerator
from sindy.quadratic_terms import SindyBasisQuadraticTermsGenerator
from sindy.sin_terms import SindyBasisSinTermsGenerator
from sindy.cos_terms import SindyBasisCosTermsGenerator
from sindy.tan_terms import SindyBasisTanTermsGenerator

def generateSindyData(robot, n_trials, n_samples_per_u, dt):
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

def printModel(robot, coeffs, terms, threshold):
    for i in range(robot.model.stateDim()):
        which_terms = np.nonzero(coeffs[:,i] > threshold)[0]
        expr = ''
        for k in which_terms:
            expr += '{:.2f}*{} + '.format(coeffs[k,i], terms[k])
        #/
        print(f'{robot.stateNames()[i]}_dot =', expr[:-3])
    # /for i
# /printModel()

def createDynamicsPythonModule(module_name, model,
                               state_names, control_names,
                               terms, coeffs,
                               threshold=1.0e-3):
    with open(f'{module_name}.py', 'w') as f:
        print('import jax', file=f)
        print('import jax.numpy as jnp', file=f)
        print('', file=f)
        print(f'class {module_name}:', file=f)
        print('', file=f)

        print('\tdef stateDim(self):', file=f)
        print(f'\t\treturn {model.stateDim()}', file=f)
        print('\t#/', file=f)
        print('', file=f)

        print('\tdef controlDim(self):', file=f)
        print(f'\t\treturn {model.controlDim()}', file=f)
        print('\t#/', file=f)
        print('', file=f)

        print(f'\tdef dynamics(self, t, s, u):', file=f)
        n_state = len(state_names)
        n_control = len(control_names)
        print('\t\t' + ', '.join(state_names) + ' = s', file=f)
        print('\t\t' + ', '.join(control_names) + ' = u', file=f)
        for i in range(n_state):
            which_terms = np.nonzero(coeffs[:,i] > threshold)[0]
            expr = ''
            for k in which_terms:
                expr += '{:.2f}*{} + '.format(coeffs[k,i], terms[k])
            #/
            expr = expr.replace('cos(', 'jnp.cos(')
            expr = expr.replace('sin(', 'jnp.sin(')
            expr = expr.replace('tan(', 'jnp.tan(')
            print(f'\t\t{state_names[i]}_dot =', expr[:-3], file=f)
        # /for i
        print('\t\treturn jnp.array([' + ', '.join(map(lambda s: s+'_dot', state_names)) + '])', file=f)
        print('\t#/', file=f)
        print('', file=f)

        print('\tdef dynamicsJacobianWrtState(self, t,s,u):', file=f)
        print('\t\treturn jax.jacfwd(self.dynamics, 1) (t,s,u)', file=f)
        print('\t#/', file=f)
        print('', file=f)

        print('\tdef dynamicsJacobianWrtControl(self, t,s,u):', file=f)
        print('\t\treturn jax.jacfwd(self.dynamics, 2) (t,s,u)', file=f)
        print('\t#/', file=f)
        print('', file=f)

        print(f'# /class {module_name}', file=f)
        print('', file=f)

    #/ with f

# /createDynamicsPythonModule

def sindy(module_name,
          robot,
          n_control_samples, n_state_samples_per_control, dt,
          threshold, verbose=False):

    # generate data and basis function values
    t_data, s_data, u_data, s_dot_data = \
        generateSindyData(robot, n_control_samples, n_state_samples_per_control, dt)
    B, basis_gens = generateSindyBasisFunctions(t_data, s_data, u_data)
    indices = np.arange(B.shape[1])
    if verbose:
        print('Iter -1 n_basis = {}'.format(B.shape[1]))
    #/

    # iteratively run least-squares and filter
    for iter in range(10):
        coeffs, residuals, rank, _ = np.linalg.lstsq(B, s_dot_data)
        argwhere_above_threshold = []
        for i in range(coeffs.shape[0]):
            if np.max(np.abs(coeffs[i,:])) > threshold:
                argwhere_above_threshold.append(i)
            # /if
        # /for
        if verbose:
            print('Iter {} rank = {} max_resid = {} n_basis = {}'
                .format(iter, rank, np.max(residuals) if len(residuals) > 0 else np.nan, len(argwhere_above_threshold)))
        # /if
        # if len(argwhere_above_threshold) == len(coeffs):
        #     break
        B = B[:, argwhere_above_threshold]
        indices = indices[argwhere_above_threshold]
    # /for iter
    coeffs = coeffs[argwhere_above_threshold, :]

    if verbose:
        print('Final norm = ', np.linalg.norm(s_dot_data - B@coeffs))
        print('coeffs |min|, |max| =', np.min(np.abs(coeffs)), np.max(np.abs(coeffs)))
        print('indices =', indices)
    # /if
    assert len(indices) > 0

    # extract non-negligible terms
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

    if verbose:
        print('fn =', terms)
        printModel(robot, coeffs, terms, threshold)
    #/

    createDynamicsPythonModule(module_name, robot.model,
                               robot.stateNames(), robot.controlNames(),
                               terms, coeffs,
                               threshold)
# /sindy()

################################################################################

from diff_drive_robot import DifferentialDriveRobot
from diff_drive_robot_2 import DifferentialDriveRobot2
from bicycle_robot import BicycleRobot
from bicycle_robot_2 import BicycleRobot2

def main():
    robot = DifferentialDriveRobot(radius=15,
                                   wheel_radius=6,
                                   wheel_thickness=3)
    robot.reset(40, 40, 0)

    # robot = DifferentialDriveRobot2(radius=15,
    #                                 wheel_radius=6,
    #                                 wheel_thickness=3)
    # robot.reset(40, 40, 0, 0, 0)

    # robot = BicycleRobot(wheel_radius=20, baseline=60)
    # robot.reset(40, 40, 0)

    # robot = BicycleRobot2(wheel_radius=20, baseline=60)
    # robot.reset(40, 40, 0, 0)

    n_control_samples = 5000
    n_state_samples_per_control = 10
    dt = 0.001
    threshold = 1.0e-2 # below which coeffs are negligible

    sindy('SINDy_DiffDriveModel', robot,
          n_control_samples, n_state_samples_per_control, dt,
          threshold, verbose=True)

# /main()

if __name__ == "__main__":
    main()
