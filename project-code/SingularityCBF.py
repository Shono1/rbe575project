from cbf_toolbox.edge import Edge
from cbf_toolbox.vertex import Agent
from cbf_toolbox.dynamics import SingleIntegrator
from ArmRobot import OMXArm
import numpy as np
from jax import grad
import jax.numpy as jnp
from jax.debug import breakpoint

class SingularityCBF():
    '''CBF that prevents an arm robot agent from entering a singular configuration'''
    def __init__(self, agent: Agent, k=1, p=1, thresh=10000):
        # if not agent.dynamics is OMXArm:
        #     print(agent.dynamics)
        #     raise ValueError('Agent must be an OMX Arm')
        self.agent = agent
        self.state = np.zeros_like(agent.state)
        self.u = np.zeros_like(agent.u)
        self.dynamics = SingleIntegrator(len(agent.state))
        self.k = k
        self.p = p

        class StupidStruct:
            '''Used to pretend i have a shape'''
            def __init__(self, barrier):
                self.func = barrier

        self.shape = StupidStruct(self.barrier)


    def add_cbf(self, m):
        x = self.agent.state
        u = self.agent.u
        thresh = self.thresh
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
        return jnp.abs(jnp.linalg.det(self.agent.dynamics.jax_jacob(x[0], x[1], x[2], x[3])[0:3, 0:3])) - thresh
    
    def step(self,u=None,dt=0.1):
        """Move forward one time step"""
        if u is None:
            u = np.zeros(self.dynamics.m)
        self.state = self.dynamics.step(self.state,u,dt)