from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    pi_host = 'team9camera@172.20.10.2'

    # ---------------------------------------------------------------
    # Launch arguments
    #
    # num_cameras: how many cameras to use (1-4)
    # test_mode:   'true'  = use local test images, no Pi needed
    #              'false' = use real Pi cameras (default)
    # image_dir:   folder containing camera_0.jpg ... camera_N.jpg
    #              (only used when test_mode:=true)
    #
    # Usage examples:
    #   Real Pi:   ros2 launch ocr_processor launch.py
    #   Test mode: ros2 launch ocr_processor launch.py test_mode:=true
    #   2 cameras: ros2 launch ocr_processor launch.py num_cameras:=2 test_mode:=true
    #   Custom dir: ros2 launch ocr_processor launch.py test_mode:=true image_dir:=/home/user/imgs
    # ---------------------------------------------------------------
    num_cameras_arg = DeclareLaunchArgument(
        'num_cameras',
        default_value='4',
        description='Number of cameras (1-4)'
    )
    test_mode_arg = DeclareLaunchArgument(
        'test_mode',
        default_value='false',
        description='true = use local test images, false = use real Pi'
    )
    image_dir_arg = DeclareLaunchArgument(
        'image_dir',
        default_value='~/test_images',
        description='Folder containing camera_0.jpg ... camera_N.jpg (test mode only)'
    )

    num_cameras = LaunchConfiguration('num_cameras')
    test_mode = LaunchConfiguration('test_mode')
    image_dir = LaunchConfiguration('image_dir')

    return LaunchDescription([
        num_cameras_arg,
        test_mode_arg,
        image_dir_arg,

        # Step 1 — Start database matcher node first
        Node(
            package='ocr_processor',
            executable='database_matcher_node',
            name='database_matcher',
            output='screen',
        ),

        # Step 2 — Start OCR node with num_cameras parameter
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='ocr_processor',
                    executable='ocr_node',
                    name='ocr_node',
                    output='screen',
                    parameters=[{'num_cameras': num_cameras}],
                ),
            ]
        ),

        # Step 3a — TEST MODE: use local images, no Pi SSH needed
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    condition=IfCondition(test_mode),
                    package='ocr_processor',
                    executable='test_publisher',
                    name='test_image_publisher',
                    output='screen',
                    parameters=[
                        {'num_cameras': num_cameras},
                        {'image_dir': image_dir},
                    ],
                ),
            ]
        ),

        # Step 3b — REAL MODE: SSH into Pi and run camera node
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    condition=UnlessCondition(test_mode),
                    cmd=[
                        'ssh', pi_host,
                        [
                            'bash -c "export ROS_DOMAIN_ID=30 && ',
                            'source /home/team9camera/ros2_humble/install/setup.bash && ',
                            'cd /home/team9camera/Project4_ws && ',
                            'source install/setup.bash && ',
                            'ros2 run camera_publisher multi_camera_node ',
                            '--ros-args -p num_cameras:=',
                            num_cameras,
                            '"'
                        ]
                    ],
                    output='screen',
                    name='camera_node',
                )
            ]
        ),
    ])