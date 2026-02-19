from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    pi_host = 'team9camera@172.20.10.2'
    
    return LaunchDescription([
        # Start OCR node first
        Node(
            package='ocr_processor',
            executable='ocr_node',
            name='ocr_processor',
            output='screen',
        ),
        
        # Wait 2 seconds for OCR to initialize, then start camera
        TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ssh', pi_host,
                        'bash -c "export ROS_DOMAIN_ID=30 && source /home/team9camera/ros2_humble/install/setup.bash && cd /home/team9camera/Project4_ws && source install/setup.bash && ros2 run camera_publisher camera_node"'
                    ],
                    output='screen',
                    name='camera_node',
                )
            ]
        ),
    ])