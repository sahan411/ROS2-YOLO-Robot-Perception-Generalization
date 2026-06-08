import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'data_collection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sdvn_hinder_metric',
    maintainer_email='sdvn_hinder_metric@todo.todo',
    description='Data collection nodes for YOLO generalization project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_driver = data_collection.robot_drive_node:main',
            'image_capture = data_collection.image_capture_node:main',
        ],
    },
)
