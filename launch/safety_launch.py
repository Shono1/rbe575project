from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get Hardware/Servo launch files from other packages
    hardware_launch_path = os.path.join(
        get_package_share_directory('open_manipulator_x_bringup'),
        'launch',
        'hardware.launch.py'
    )

    servo_launch_path = os.path.join(
        get_package_share_directory('open_manipulator_x_moveit_config'),
        'launch',
        'servo.launch.py'
    )

    return LaunchDescription([
        # Safety controller
        Node(package='safety', executable='safety_controller', output='screen'),
        # Hardware/Servo Launch (REQUIRED TO RUN ARM)
        IncludeLaunchDescription(PythonLaunchDescriptionSource(hardware_launch_path)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(servo_launch_path)),
    ])
