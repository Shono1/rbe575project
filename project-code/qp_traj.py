import jax.numpy as jnp
from cbfpy import CBF, CBFConfig
import pickle as pkl
import numpy as np
import roboticstoolbox as rtb
import sympy as sym

MAX_STEP = 0.05
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
    # The *relative degree* of this system is 2, so, we'll use the h_2 method
    def h_1(self, z):
        # print(f'z: {z}')
        # print(f'jac(z): {self.jacob(*z)}')
        # print(f'det(z): {jnp.linalg.det(self.jacob(*z))}')
        # print(jnp.log10(jnp.linalg.det(self.jacob(*z)[0:3, 0:3])))
        return jnp.array([jnp.log10(jnp.linalg.det(self.jacob(*z))) - 4])

    # define class k funciton
    # def alpha(self, h):
    #     return jnp.sqrt(h)

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

print(type(js_traj))
js_traj.extend([js_traj[-1]] * len(js_traj))  # Append final point to end of trajectory a few times
# print([pt.t for pt in js_traj])
z = js_traj[0]

# Create CBF
config = ArmCBF(jacob=lambda_jac)
cbf = CBF.from_config(config)

# Run filtered traj (OLD JOINT SPACE CODE)
# ts = []
# for i, qs in enumerate(js_traj):
#     # z = get_state()
#     z_des = qs
#     u_nom = z_des - z
#     u = cbf.safety_filter(z, u_nom)
#     z += u
#     print(z)
#     ts.append(robot.fkine(np.array(z)))
# print([pt.t for pt in ts])
# with open('ts_record_filter.pkl', 'wb') as f:
#     pkl.dump(ts, f)

# ts_traj = [robot.fkine(qs).t for qs in js_traj]
ts_traj = np.array([np.repeat([0], 1000), np.linspace(-150, 150, 1000), np.repeat([150], 1000)]).T
ts_followed = []
js_followed = []
z = js_traj[0]
for i, ts_pt in enumerate(ts_traj):
    # Calculate delta in task space
    delta_ts = ts_pt - robot.fkine(np.array(z)).t

    # Get Jacobian and invert
    # print(robot.jacob0(np.array(z)))
    inv_jac = np.linalg.pinv(robot.jacob0(np.array(z), half='trans'))
    u_nom = np.clip(inv_jac @ delta_ts, -MAX_STEP, MAX_STEP)

    # Get control and update robot state
    u = cbf.safety_filter(z, u_nom)
    if any(jnp.isnan(u)):
        print('skipping nan nan nan nan nan nan nan nan')
        continue
    z += u 
    print(u)
    ts_followed.append(robot.fkine(np.array(z)).t)
    js_followed.append(z)

with open('ts_record_filter.pkl', 'wb') as f:
    pkl.dump(ts_followed, f)

with open('js_record_filtered.pkl', 'wb') as f:
    pkl.dump(js_followed, f)