"""Launch oren_ros mapping_node (+ sdf_query_node, RViz) for LIVE sensor topics.

mapping.launch.py から次の点だけを変えた実機用の launch ファイル:
  - `ros2 bag play` と clock_node を起動しない（実機では /clock も bag も不要）
  - use_sim_time の既定値を false にする
  - ROS 側のパラメータファイルを引数 params_file で差し替えられるようにする

ROS 側のパラメータ（トピック名、modality、sdf_query_node のグリッドなど）は
configs/ros2-params.yaml（または params_file で指定したファイル）に書く。

使い方:
  ros2 launch oren_ros live.launch.py \
      trainer_config_path:=/path/to/my_trainer.yaml
  # ROS 側のパラメータを自分のファイルにしたい場合
  ros2 launch oren_ros live.launch.py \
      trainer_config_path:=/path/to/my_trainer.yaml params_file:=/path/to/my_ros2_params.yaml
  # 保存済みの final.pth から続きを作る（省略すると、これまでどおり 1 から作る）
  ros2 launch oren_ros live.launch.py \
      trainer_config_path:=/path/to/my_trainer.yaml init_model_path:=/path/to/final.pth
  # sdf_query_node も RViz も使わない（GPU の負荷を下げる）
  ros2 launch oren_ros live.launch.py trainer_config_path:=... visualize_sdf:=false rviz:=false
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _rviz_action(context, *args, **kwargs):
    """Resolve rviz_config at runtime and launch RViz2 with the specified config."""
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    rviz_args = ["-d", rviz_config] if rviz_config else []

    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=rviz_args,
            condition=IfCondition(LaunchConfiguration("rviz")),
        )
    ]


def generate_launch_description():
    oren_ros_share = get_package_share_directory("oren_ros")

    trainer_config_path_arg = DeclareLaunchArgument(
        "trainer_config_path",
        default_value=PathJoinSubstitution([oren_ros_share, "configs", "trainer-ros.yaml"]),
        description="Trainer config YAML path.",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([oren_ros_share, "configs", "ros2-params.yaml"]),
        description="ROS 2 parameter YAML for oren_mapping_node / sdf_query_node.",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Keep false for live sensors. true only when /clock is published by a bag player.",
    )

    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="If true, launch RViz alongside the mapping node.",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=PathJoinSubstitution([oren_ros_share, "rviz", "lidar.rviz"]),
        description="Path to the RViz configuration file.",
    )

    visualize_sdf_arg = DeclareLaunchArgument(
        "visualize_sdf",
        default_value="true",
        description="If true, launch sdf_query_node to publish SDF as GridMap / PointCloud2.",
    )

    init_model_path_arg = DeclareLaunchArgument(
        "init_model_path",
        default_value="",
        description="Path to a saved final.pth to resume from. Empty (default) = build the map from scratch.",
    )

    trainer_config_path = LaunchConfiguration("trainer_config_path")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    mapping_node = Node(
        package="oren_ros",
        executable="mapping_node",
        name="oren_mapping_node",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": use_sim_time},
            {"trainer_config_path": trainer_config_path},
            # 空文字でも文字列のまま渡す（value_type=str が無いと、空文字が yaml で None になりエラーになる）
            {"init_model_path": ParameterValue(LaunchConfiguration("init_model_path"), value_type=str)},
        ],
    )

    sdf_query_node = Node(
        package="oren_ros",
        executable="sdf_query_node",
        name="sdf_query_node",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration("visualize_sdf")),
    )

    return LaunchDescription(
        [
            trainer_config_path_arg,
            params_file_arg,
            use_sim_time_arg,
            rviz_arg,
            rviz_config_arg,
            visualize_sdf_arg,
            init_model_path_arg,
            LogInfo(
                msg=[
                    "Starting oren_ros LIVE mapping with trainer config: ",
                    trainer_config_path,
                    " | params: ",
                    params_file,
                    " | use_sim_time: ",
                    use_sim_time,
                ]
            ),
            mapping_node,
            sdf_query_node,
            OpaqueFunction(function=_rviz_action),
        ]
    )
