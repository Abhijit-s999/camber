"""Bridge plus example pilot. Override the vehicle with spec:=path/to/vehicle."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))


def generate_launch_description():
    spec = LaunchConfiguration("spec")
    viewer = LaunchConfiguration("viewer")
    return LaunchDescription([
        DeclareLaunchArgument("spec", default_value=os.path.join(
            os.environ.get("CAMBER_ROOT", REPO), "examples", "racer5.toml")),
        DeclareLaunchArgument("viewer", default_value="true"),
        Node(package="camber_ros", executable="bridge", name="camber_sim",
             output="screen",
             parameters=[{"spec": spec, "viewer": viewer, "course": "oval"}]),
        Node(package="camber_ros", executable="gate_pilot", name="gate_pilot",
             output="screen", parameters=[{"use_sim_time": True, "laps": 2}]),
    ])
