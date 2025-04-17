from ArmRobot import OMXArm
from SingularityCBF import SingularityCBF
import roboticstoolbox as rtb
import numpy as np
from cbf_toolbox.vertex import Agent, Goal
from cbf_toolbox.geometry import Point
from cbf_toolbox.safety import Simulation
from cbf_toolbox.dynamics import SingleIntegrator
from time import sleep

dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
robot = rtb.DHRobot(dh_tab, name='OMX-Arm')
# robot.plot([0, 0, 0, 0], block=True)
dyn = OMXArm(robot)

agent = Agent(np.array([0, 0, 0, 0]), Point(4), dyn)
control = Goal(np.array([0, 0, 0, 0]), shape=Point(4), dynamics=SingleIntegrator(4))
s = Simulation()
s.add_agent(agent, control)
# s.simulate(num_steps=100, dt = 0.1)
robot.plot(agent.state)
for i in range(100):
    s.step(0.1)
    # print(agent.state)
    robot.plot(agent.state)
    # sleep(0.1)
    