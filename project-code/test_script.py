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

# Create trajectory generator

# Add everything to simulation
agent = Agent(np.array([0.1, -np.pi/2, 0.2, -0.5]), Point(4), dyn)
control = Goal(np.array([0, 0, 0, 0]), shape=Point(4), dynamics=SingleIntegrator(4))
s = Simulation()
s.add_agent(agent, control)
s.add_obstacle(SingularityCBF(agent))

# Simulate
robot.plot(agent.state)
for i in range(100):
    s.step(0.1)
    J = robot.jacob0(agent.state)
    print(f' state: {agent.state}   manip: {jnp.linalg.det(lambda_jac(*agent.state)[0:3, 0:3]) - 10}')
    robot.plot(agent.state)