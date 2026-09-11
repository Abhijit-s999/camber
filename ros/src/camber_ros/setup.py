from glob import glob

from setuptools import setup

setup(
    name="camber_ros",
    version="0.1.0",
    packages=["camber_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/camber_ros"]),
        ("share/camber_ros", ["package.xml"]),
        ("share/camber_ros/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "bridge = camber_ros.bridge:main",
        "gate_pilot = camber_ros.gate_pilot:main",
    ]},
)
