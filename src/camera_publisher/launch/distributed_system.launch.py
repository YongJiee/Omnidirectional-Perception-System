from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    #pi_host = 'team9camera@172.20.10.2' #"WIFI"
    pi_host = 'team9camera@192.168.10.1' #Ethernet

    # ---------------------------------------------------------------
    # Launch arguments
    #
    # num_cameras: how many cameras to use (1-4)
    # test_mode:   'true'  = use local test images, no Pi needed
    #              'false' = use real Pi cameras (default)
    # image_dir:   folder containing camera_0.jpg ... camera_N.jpg
    #              (only used when test_mode:=true)
    # scan_mode:   'inbound'  = collect all faces then match once
    #              'sorting'  = match as fast as possible (default)
    #
    # Usage examples:
    #   Real Pi (sorting):  ros2 launch camera_publisher distributed_system.launch.py num_cameras:=2
    #   Real Pi (inbound):  ros2 launch camera_publisher distributed_system.launch.py num_cameras:=2 scan_mode:=inbound
    #   Test mode:          ros2 launch camera_publisher distributed_system.launch.py test_mode:=true scan_mode:=inbound
    #   Custom dir:         ros2 launch camera_publisher distributed_system.launch.py test_mode:=true image_dir:=/home/user/imgs
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
    scan_mode_arg = DeclareLaunchArgument(
        'scan_mode',
        default_value='sorting',
        description='inbound = collect all faces then match once | sorting = match as fast as possible'
    )

    num_cameras = LaunchConfiguration('num_cameras')
    test_mode = LaunchConfiguration('test_mode')
    image_dir = LaunchConfiguration('image_dir')
    scan_mode = LaunchConfiguration('scan_mode')

    return LaunchDescription([
        num_cameras_arg,
        test_mode_arg,
        image_dir_arg,
        scan_mode_arg,          # FIX: was declared but never added here

        # Step 1 — Start database matcher node first
        Node(
            package='ocr_processor',
            executable='database_matcher_node',
            name='database_matcher',
            output='screen',
            parameters=[
                {'scan_mode': scan_mode},
            ],
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
                    parameters=[
                        {'num_cameras': num_cameras},
                        {'scan_mode': scan_mode},
                    ],
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
                        {'scan_mode': scan_mode},
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
                            '/home/team9camera/start_camera.sh ',
                            num_cameras,
                            ' ',
                            scan_mode,
                        ]
                    ],
                    output='screen',
                    name='camera_node',
                )
            ]
        ),
    ])