from glob import glob
import os

from setuptools import find_packages, setup


package_name = "oren_ros"


def _collect_data_files(directory):
    entries = []

    if not os.path.isdir(directory):
        return entries

    for dirpath, _, filenames in os.walk(directory, followlinks=True):
        files = [
            os.path.join(dirpath, filename)
            for filename in filenames
        ]

        if files:
            destination = os.path.join(
                "share",
                package_name,
                dirpath,
            )
            entries.append((destination, files))

    return entries


setup(
    name=package_name,
    version="0.1.0",
    author="Zhirui Dai, Qihao Qian",
    author_email="zhdai@ucsd.edu, q2qian@ucsd.edu",
    description="ROS 2 wrapper for the oren algorithm.",
    url="https://github.com/ExistentialRobotics/oren",
    license="MIT",
    install_requires=["setuptools"],
    packages=find_packages(),
    zip_safe=True,
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            "share/" + package_name + "/launch",
            glob("launch/*.py"),
        ),
        *_collect_data_files("configs"),
        *_collect_data_files("rviz"),
    ],
    entry_points={
        "console_scripts": [
            "mapping_node = oren_ros.node.mapping_node:main",
            "clock_node = oren_ros.node.clock_node:main",
            "sdf_query_node = oren_ros.node.sdf_query_node:main",
        ],
    },
)
