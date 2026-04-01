from setuptools import find_packages, setup

package_name = "beavr"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test", "test.*", "*.test", "*.tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="BeaVR Team",
    maintainer_email="team@beavr.ai",
    description="BeaVR-Bot: Bimanual, multi-Embodiment, Accessible, Virtual Reality Teleoperation System for Robots",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
