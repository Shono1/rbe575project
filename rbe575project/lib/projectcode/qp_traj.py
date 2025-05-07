import jax.numpy as jnp
from cbfpy import CBF, CBFConfig
import pickle as pkl
import numpy as np
import roboticstoolbox as rtb
import sympy as sym
from copy import deepcopy
from spatialmath import SE3

MAX_STEP = 0.1
DET_THRESH = 100  # Keep jacobian determinant above this.
TIMESTEPS = 500
Q1_MIN = -np.pi 
Q1_MAX = np.pi 
Q2_MIN = -np.pi / 2
Q2_MAX = np.pi / 2
Q3_MIN = -np.pi / 2
Q3_MAX = np.pi / 2
Q4_MIN = -9 * np.pi / 16
Q4_MAX = 10 * np.pi / 16

Z_HEIGHT = 220
Y_START = -270
Y_GOAL = 270

# MAX_STEP = 0.025
# Create a config class for your problem inheriting from the CBFConfig class
class ArmCBF(CBFConfig):
    def __init__(self, jacob):
        self.jacob = jacob
        super().__init__(
            # Define the state and control dimensions
            n = 4, # [x, x_dot]
            m = 4, # [F_x]
            # Define control limits (if desired)
            u_min = [-MAX_STEP]*4,
            u_max = [MAX_STEP]*4)
        

    # Define the control-affine dynamics functions `f` and `g` for your system
    def f(self, z):
        # No drift dynamics
        return z

    def g(self, z):
        # Single integrator
        B = jnp.eye(4)
        return B

    # Define the barrier function `h`
    def h_1(self, z):
        if DET_THRESH == -1:
            return jnp.array([
            Q1_MAX - z[0],
            z[0] - Q1_MIN,
            Q2_MAX - z[1],
            z[1] - Q2_MIN,
            Q3_MAX - z[2],
            z[2] - Q3_MIN, 
            Q4_MAX - z[3],
            z[3] - Q4_MIN]) 
        return jnp.array([
            jnp.log10(jnp.linalg.det(self.jacob(*z))) - DET_THRESH,  # Avoid singularities
            Q1_MAX - z[0],
            z[0] - Q1_MIN,
            Q2_MAX - z[1],
            z[1] - Q2_MIN,
            Q3_MAX - z[2],
            z[2] - Q3_MIN, 
            Q4_MAX - z[3],
            z[3] - Q4_MIN]) 
    
def calc_det(z):
    # Calculate the CBF values for the given state z without any control
    return jnp.array(
        jnp.log10(jnp.linalg.det(lambda_jac(*z)))# Avoid singularities
        ) 
            

# Build robot object
dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')
symbot = rtb.DHRobot(dh_tab, name='OMX-arm-sym', symbolic=True)
th1, th2, th3, th4 = sym.symbols('th1 th2 th3 th4')
print('Calculating Jacobian')
sym_jac = symbot.jacob0([th1, th2, th3, th4])
print('Simplifying Jacobian')
sym_jac = sym_jac[0:3, 0:3]
# print(sym_jac.shape)
sym_jac = sym.Matrix(sym_jac)
# sym_jac = sym.simplify(sym_jac)
print('Lambdifying Jacobian')
lambda_jac = sym.lambdify([th1, th2, th3, th4], sym_jac, 'jax')

# Load trajectory
with open('js_traj.pkl', 'rb') as f:
    js_traj = pkl.load(f)

js_traj.extend([js_traj[-1]] * len(js_traj))  # Append final point to end of trajectory a few times
# print([pt.t for pt in js_traj])
z = js_traj[0]

# Create CBF
config = ArmCBF(jacob=lambda_jac)
cbf = CBF.from_config(config)

# ts_traj = [robot.fkine(qs).t for qs in js_traj]
ts_traj = np.array([np.repeat([0], TIMESTEPS), np.linspace(Y_START, Y_GOAL, TIMESTEPS), np.repeat([Z_HEIGHT], TIMESTEPS)]).T
ts_followed = []
js_followed = []
cbf_followed = []
# z = js_traj[-1]
start = SE3(0, Y_START, Z_HEIGHT)
z = robot.ik_LM(start, mask=[1, 1, 1, 0, 0, 0], q0=[-np.pi/2, 0, 0, 0])[0]
for i, ts_pt in enumerate(ts_traj):
    # Calculate delta in task space
    ts_pt = ts_traj[-1]
    delta_ts = ts_pt - robot.fkine(np.array(z)).t
    # print(np.linalg.norm(delta_ts))
    # Get Jacobian and invert
    # print(robot.jacob0(np.array(z)))
    inv_jac = np.linalg.pinv(robot.jacob0(np.array(z), half='trans'))
    # u_nom = np.clip(inv_jac @ delta_ts, -MAX_STEP, MAX_STEP)
    u_nom = inv_jac @ delta_ts
    # Normalize to have max magnitude of MAX_STEP
    # u_nom *= np.clip(1, )
    if norm := np.linalg.norm(u_nom) > MAX_STEP:
        u_nom = (u_nom / norm) * MAX_STEP 

    # Get control and update robot state
    # u = cbf.safety_filter(z, u_nom)
    u = u_nom
    noise = np.random.normal(0, 0.00002, (4,)) 
    # print(noise)
    u += noise
    if norm := np.linalg.norm(u) > MAX_STEP:
        u = (u / norm) * MAX_STEP 
    
    if any(jnp.isnan(u)):
        print('skipping nan nan nan nan nan nan nan nan')
        continue
    z += u 

    ts_followed.append(robot.fkine(np.array(z)).t)
    js_followed.append(deepcopy(z))
    cbf_followed.append(np.array(calc_det(z)))

with open(f'rbe575project/lib/projectcode/ts_traj/ts_traj_{DET_THRESH}.pkl', 'wb') as f:
    pkl.dump(ts_followed, f)

with open(f'rbe575project/lib/projectcode/js_traj/js_traj_{DET_THRESH}.pkl', 'wb') as f:
    pkl.dump(js_followed, f)

with open(f'rbe575project/lib/projectcode/cbf_functions/cbf_functions_{DET_THRESH}.pkl', 'wb') as f:
    pkl.dump(cbf_followed, f)