import rclpy
from rclpy.node import Node
# from .ArmRobot import OMXArm
# from .SingularityCBF import SingularityCBF
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
import pickle as pkl
from std_msgs.msg import Bool
import os, time


from std_msgs.msg import Float64MultiArray

class CBFNode(Node):
    def __init__(self):
        super().__init__('cbf_node')

        # Build robot object
        # dh_tab = [rtb.DHLink(d=96.326, alpha=-np.pi/2, a=0, offset=0, qlim=[-np.pi, np.pi]),
        #             rtb.DHLink(d=0, alpha=0, a=np.sqrt(24**2 + 128**2), offset=-np.arctan2(128, 24), qlim=[-np.pi/1.5, np.pi/1.5]),
        #             rtb.DHLink(d=0, alpha=0, a=124, offset=np.arctan2(128, 24), qlim=[-np.pi/1.5, np.pi/1.5]),
        #             rtb.DHLink(d=0, alpha=0, a=133.4, offset=0, qlim=[-np.pi/1.5, np.pi/1.5])]
        # self.robot = rtb.DHRobot(dh_tab, name='OMX-Arm')

        # # Setup trajectory in task space
        # start = SE3(0, 150, 150)
        # end = SE3(0, -150, 150)
        # t = np.arange(0, 4, 0.1)
        # self.ts_traj = rtb.ctraj(start, end, t)

        # # Guess joint space trajectory
        # guess = self.robot.ik_LM(self.ts_traj[0], mask=[1, 1, 1, 0, 0, 0])[0]
        # self.js_traj = [guess]

        # Get the directory of the current script

        with open('/home/jhkeselman/colcon_ws/src/rbe575project/rbe575project/lib/projectcode/js_traj_nocbf.pkl', 'rb') as f:
        # with open('/home/cooper530/rbe575/robot/src/rbe575project/rbe575project/lib/projectcode/js_traj_nocbf.pkl', 'rb') as f:
            self.js_traj = pkl.load(f)

        with open('/home/jhkeselman/colcon_ws/src/rbe575project/rbe575project/lib/projectcode/ts_traj_nocbf.pkl', 'rb') as f:
        # with open('/home/cooper530/rbe575/robot/src/rbe575project/rbe575project/lib/projectcode/ts_traj_nocbf.pkl', 'rb') as f:
            self.ts_traj = pkl.load(f)

        # Setup publisher and subscriber
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_positions', 10)

        self.position_pub = self.create_publisher(Float64MultiArray, '/task_position', 10)

        self.create_subscription(Bool, '/starttraj', self.update_position, 10)

        self.get_logger().info("CBF Node has been initialized.")
    
    def update_position(self, msg):
        if(msg.data):
            for joints in self.js_traj:
                send = Float64MultiArray()
                send.data = joints.tolist()
                self.joint_pub.publish(send)
                time.sleep(0.1)
        else:
            for pos in self.ts_traj:
                send = Float64MultiArray()
                send.data = pos.t.tolist()
                self.position_pub.publish(send)
                time.sleep(0.1)


    def destroy_node(self):
        self.get_logger().info('Destroying node...')
        super().destroy_node()
    
def main():
    rclpy.init()
    cbf_node = CBFNode()
    rclpy.spin(cbf_node)
    cbf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()