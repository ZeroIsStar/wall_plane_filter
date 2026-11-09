from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('wall_plane_filter')
    default_params = os.path.join(pkg_share, 'config', 'wall_plane_filter.yaml')
    # Prefer the workspace DDS profile shared with WSL bag/RViz. /capella/dds.xml
    # binds docker0 which is down here, so filtered clouds never reach RViz.
    dds_uri = os.environ.get('CYCLONEDDS_URI', 'file:///workspace/dds_local.xml')

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
        SetEnvironmentVariable('CYCLONEDDS_URI', dds_uri),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        Node(
            package='wall_plane_filter',
            executable='wall_plane_filter_node',
            name='wall_plane_filter',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
            additional_env={
                'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
                'CYCLONEDDS_URI': dds_uri,
                'ROS_LOCALHOST_ONLY': '0',
            },
        ),
    ])
