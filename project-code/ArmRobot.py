import roboticstoolbox as rtb
from cbf_toolbox.dynamics import Dynamics
import numpy as np

class OMXArm(Dynamics):
    def __init__(self, robot: rtb.DHRobot):
        '''Initiailizes robot dynamnics from a DH table'''
        self.robot = robot
        n = robot.n
        m = robot.n
        f = 0
        g = np.eye(n)
        super().__init__(n, m, f, g)


if __name__ == '__main__':
    dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
    robot = rtb.DHRobot(dh_tab, name='OMX-Arm')
    # robot.plot([0, 0, 0, 0], block=True)
    dyn = OMXArm(robot)
