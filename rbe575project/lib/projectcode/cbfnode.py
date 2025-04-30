import rclpy
from rclpy.node import Node

class CBFNode(Node):
    def __init__(self):
        super().__init__('cbf_node')
        self.get_logger().info("CBF Node has been initialized.")
    
    def destroy_node(self):
        self.get_logger().info('Destroying node...')
        super().destroy_node()
    
def main():
    rclpy.init()
    cbf_node = CBFNode()
    rclpy.spin(cbf_node)
    cbf_node.stop_moveit_servo()
    cbf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()