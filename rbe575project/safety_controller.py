import rclpy, time
from rclpy.node import Node
from control_msgs.msg import JointJog
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class SafetyController(Node):
    def __init__(self):
        super().__init__('safety_controller')

        self.servo_start_client = self.create_client(Trigger, '/servo_node/start_servo')
        self.servo_stop_client = self.create_client(Trigger, '/servo_node/stop_servo')

        # Connect to services
        self.connect_moveit_servo()

        # Example usage
        self.start_moveit_servo()

        # Create publisher to command robot
        self.joint_pub = self.create_publisher(JointJog, '/servo_node/delta_joint_cmds', 10)

        self.pos_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        # Subscribe to joint velocities
        self.create_subscription(Float64MultiArray, '/joint_velocities', self.velo_callback, 10)

        # Subscribe to joint positions, this is how the CBF component will send commands
        self.create_subscription(Float64MultiArray, '/joint_positions', self.joint_callback, 10)

        self.create_subscription(Float64MultiArray, '/task_position', self.pos_callback, 10)

        self.joint_positions = [0,0,0,0]
        self.ee_position = []

        # Subscribe to save joint positions
        self.create_subscription(JointState, '/joint_states', self.update_current_positions, 10)

    def update_current_positions(self, msg):
        names = msg.name
        positions = msg.position
        for i in range(len(names)):
            joint = names[i]
            if joint == "joint1":
                self.joint_positions[0] = positions[i]
            elif joint == "joint2":
                self.joint_positions[1] = positions[i]
            elif joint == "joint3":
                self.joint_positions[2] = positions[i]
            elif joint == "joint4":
                self.joint_positions[3] = positions[i]

    def joint_callback(self, msg):
        self.get_logger().info('Sending joint positions...')
        joint_jog = JointJog()
        joint_jog.header.stamp = self.get_clock().now().to_msg()
        joint_jog.header.frame_id = 'link1'
        # Add joint names
        joint_jog.joint_names = ["joint1", "joint2", "joint3", "joint4"]
        
        # Get displacement
        duration = 0.1
        velocities = [(b_i - a_i) / duration for a_i, b_i in zip(self.joint_positions, msg.data)]
        print(velocities)

        # Add velocities to JointJog
        joint_jog.velocities = velocities
        # Publish JointJog
        self.joint_pub.publish(joint_jog)

    def pos_callback(self, msg):
        # self.get_logger().info('Sending task positions...')
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'link1'


        # If the list is empty update it's starting position
        new_pos = [ele / 1000.0 for ele in msg.data]

        if not self.ee_position:
            self.ee_position = new_pos

        # Get displacement (task positions)
        duration = 0.1
        velocities = [(b_i - a_i) / duration for a_i, b_i in zip(self.ee_position, new_pos)]

        for i in range(len(velocities)):
            if velocities[i] >= 1.0:
                velocities[i] = 0.9
        
        for i in range(len(velocities)):
            if velocities[i] <= -1.0:
                velocities[i] = -0.9
        
        self.get_logger().info(str(velocities))

        twist.twist.linear.x = velocities[0]
        twist.twist.linear.y = velocities[1]
        twist.twist.linear.z = velocities[2]

        # Update current end effector position
        self.ee_position = new_pos

        self.pos_pub.publish(twist)

    

    def velo_callback(self, msg):
        self.get_logger().info('Sending joint velocities...')
        joint_jog = JointJog()
        joint_jog.header.stamp = self.get_clock().now().to_msg()
        joint_jog.header.frame_id = 'link1'

        # Add joint names
        joint_jog.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        # Add velocities to JointJog
        joint_jog.velocities = msg.data

        # Publish JointJog
        self.joint_pub.publish(joint_jog)

    def connect_moveit_servo(self):
        for i in range(10):
            if self.servo_start_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("SUCCESS TO CONNECT SERVO START SERVER")
                break
            self.get_logger().warn("WAITING TO CONNECT SERVO START SERVER...")
            if i == 9:
                self.get_logger().error("FAILED to connect to moveit_servo. Please launch 'servo.launch'")

        for i in range(10):
            if self.servo_stop_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("SUCCESS TO CONNECT SERVO STOP SERVER")
                break
            self.get_logger().warn("WAITING TO CONNECT SERVO STOP SERVER...")
            if i == 9:
                self.get_logger().error("FAILED to connect to moveit_servo. Please launch 'servo.launch'")

    def start_moveit_servo(self):
        self.get_logger().info("Calling 'moveit_servo' start service...")
        req = Trigger.Request()
        future = self.servo_start_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result():
            self.get_logger().info("SUCCESS to start 'moveit_servo'")
        else:
            self.get_logger().error("FAILED to start 'moveit_servo'")

    def stop_moveit_servo(self):
        self.get_logger().info("Calling 'moveit_servo' stop service...")
        req = Trigger.Request()
        future = self.servo_stop_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result():
            self.get_logger().info("SUCCESS to stop 'moveit_servo'")
        else:
            self.get_logger().error("FAILED to stop 'moveit_servo'")
         
    
    def destroy_node(self):
        self.get_logger().info('Destroying node...')
        super().destroy_node()

def main():
    rclpy.init()
    arm_controller = SafetyController()
    rclpy.spin(arm_controller)
    arm_controller.stop_moveit_servo()
    arm_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
