import rclpy
from rclpy.node import Node
from control_msgs.msg import JointJog
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

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
        
        # Subscribe to joint velocities
        self.create_subscription(Float64MultiArray, '/joint_velocities', self.velo_callback, 10)

        # Subscribe to joint positions, this is how the CBF component will send commands
        self.create_subscription(Float64MultiArray, '/joint_positions', self.pos_callback, 10)

    def pos_callback(self, msg):
        self.get_logger().info('Sending joint positions...')
        joint_jog = JointJog()
        joint_jog.header.stamp = self.get_clock().now().to_msg()
        joint_jog.header.frame_id = 'link1'

        # Add joint names
        joint_jog.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        # Add velocities to JointJog
        joint_jog.displacements = msg.data

        # Publish JointJog
        self.joint_pub.publish(joint_jog)
    

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
