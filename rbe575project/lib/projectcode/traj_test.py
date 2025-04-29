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
from spatialmath import SE3
from spatialmath.base.transforms3d import trplot
import matplotlib.pyplot as plt
import pickle as pkl

# Build robot object
dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0, qlim=[-np.pi, np.pi]),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24), qlim=[-np.pi/1.5, np.pi/1.5]),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24), qlim=[-np.pi/1.5, np.pi/1.5]),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0, qlim=[-np.pi/1.5, np.pi/1.5])]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')

start = SE3(0, 150, 150)
end = SE3(0, -150, 150)
t = np.arange(0, 4, 0.1)
ts_traj = rtb.ctraj(start, end, t)
# js_traj = [robot.ik_LM(pt, mask=[1, 1, 1, 0, 0, 0])[0] for pt in ts_traj]
# print(js_traj)
# converrt from task to joint space
guess = robot.ik_LM(ts_traj[0], mask=[1, 1, 1, 0, 0, 0])[0]
js_traj = [guess]
for i, pt in enumerate(ts_traj[1:]):
    res = robot.ik_LM(pt, mask=[1, 1, 1, 0, 0, 0], q0=js_traj[i], joint_limits=True)
    print(res)
    js_traj.append(res[0])

print(js_traj)

fig = robot.plot(js_traj[0])
for qs in js_traj:
    fig = robot.plot(qs)

with open('js_traj.pkl', 'wb') as f:
    pkl.dump(js_traj, f)


# ikdata = robot.ik_LM(start, mask=[1, 1, 1, 0, 0, 0])
# print(ikdata)
# joints = ikdata[0]
# print(joints)

# robot.plot(joints, block=True)
# breakpoint()
# start_t = SE3(0, -0.2, 150)