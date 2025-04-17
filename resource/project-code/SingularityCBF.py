from cbf_toolbox.edge import Edge
from cbf_toolbox.vertex import Agent
from ArmRobot import OMXArm
import numpy as np
from jax import grad

class SingularityCBF():
    '''CBF that prevents an arm robot agent from entering a singular configuration'''
    def __init__(self, agent: Agent, k=1, p=1):
        if not agent.dynamics is OMXArm:
            raise ValueError('Agent must be an OMX Arm')
        self.agent = agent
        self.k = k
        self.p = p

    def add_cbf(self, m):
        x = self.agent.state
        u = self.agent.u
        thresh = 10
        h = self.barrier(x, thresh)
        grad_h = np.array(grad(self.barrier, argnums=0)(x, thresh))
        xdot = self.agent.dynamics.dx(x, u)

        for i in range(2):         
            # Try pushing x slightly (repeat 2 times in different directions)   
            if np.isnan(grad_h[0]):
                x[i] += .001
                grad_h = np.array(grad(self.barrier, argnums=0)(x, thresh))
            else:
                break

        lg_h = grad_h.T.dot(xdot)

        m.addConstr((lg_h)>=-self.k*h**self.p, "cbf")
        m.update()

    def barrier(self, x, thresh):
        '''Calculates the manipulability of the robot'''
        return self.agent.dynamics.robot.manipulability(x) - thresh