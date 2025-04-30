from ArmRobot import OMXArm
from SingularityCBF import SingularityCBF
import roboticstoolbox as rtb
import numpy as np
from cbf_toolbox.vertex import Agent, Goal
from cbf_toolbox.geometry import Point
from cbf_toolbox.safety import Simulation
from cbf_toolbox.dynamics import SingleIntegrator
from time import sleep
import jax.numpy as jnp
import sympy as sym
import pickle as pkl
# import dill
# dill.settings['recurse'] = True
import sys
# print(sys.setrecursionlimit(10_000))

# Build robot object
dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')
symbot = rtb.DHRobot(dh_tab, name='OMX-arm-sym', symbolic=True)
th1, th2, th3, th4 = sym.symbols('th1 th2 th3 th4')
sym_jac = sym.simplify(symbot.jacob0([th1, th2, th3, th4]))
lambda_jac = sym.lambdify([th1, th2, th3, th4], sym_jac, 'jax')
dyn = OMXArm(robot, lambda_jac)

# with open('lambda_jac.dill', 'wb') as f:
#     dill.dump(lambda_jac, f)

# Add everything to simulation
with open('js_traj.pkl', 'rb') as f:
    js_traj = pkl.load(f)

js_traj.extend([js_traj[-1] * len(js_traj) * 10])

agent = Agent(np.array(js_traj[0]), Point(4), dyn, p=1000)
# control = Goal(np.array(js_traj[0]), shape=Point(4), dynamics=SingleIntegrator(4), gamma=1, p=100)
control = Goal(np.array(js_traj[0]), shape=Point(4), dynamics=SingleIntegrator(4))

s = Simulation()
s.add_agent(agent, control, upper_bounds=0.5, lower_bounds=-0.5)
s.add_obstacle(SingularityCBF(agent, thresh=0, k=100))

# Simulate
robot.plot(agent.state)
qs = []
ts = []
for i in range(len(js_traj)):
    s.step(0.1)
    # J = robot.jacob0(agent.state)
    control.state = js_traj[i]
    print(f' state: {agent.state}   manip: {np.log10(jnp.linalg.det(lambda_jac(*agent.state)[0:3, 0:3]))}')
    # robot.plot(agent.state)
    
    qs.append(agent.state)
    ts.append(robot.fkine(agent.state))

print(ts)
with open('ts_record.pkl', 'wb') as f:
    pkl.dump(ts, f)

with open('js_record.pkl', 'wb') as f:
    pkl.dump(qs, f)