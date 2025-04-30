import roboticstoolbox as rtb
from cbf_toolbox.dynamics import Dynamics
import numpy as np
import sympy as sym

class OMXArm(Dynamics):
    def __init__(self, robot: rtb.DHRobot, jax_jacob: np.ndarray):
        '''Initiailizes robot dynamics from a DH table'''
        self.robot = robot
        n = robot.n
        m = robot.n
        f = lambda x: 0
        g = lambda x: np.eye(n)
        self.jax_jacob = jax_jacob
        super().__init__(n, m, f, g)


if __name__ == '__main__':
    dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0),
              rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24)),
              rtb.DHLink(d=0, alpha=0, a=133.4, offset=0)]
    robot = rtb.DHRobot(dh_tab, name='OMX-Arm')
    symbot = rtb.DHRobot(dh_tab, name='OMX-Arm-sym', symbolic=True)
    th1, th2, th3, th4 = sym.symbols('th1 th2 th3 th4')
    sym_jac = sym.simplify(symbot.jacob0([th1, th2, th3, th4]))
    lambda_jac = sym.lambdify([th1, th2, th3, th4], sym_jac, 'numpy')
    print(lambda_jac(0, 0, 0, 0))
    # robot.plot([0, 0, 0, 0], block=True)
    dyn = OMXArm(robot, lambda_jac)
