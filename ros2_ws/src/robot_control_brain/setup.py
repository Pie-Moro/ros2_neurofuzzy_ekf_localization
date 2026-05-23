from setuptools import find_packages, setup

package_name = 'robot_control_brain'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'torch',
        'scipy'
    ],
    zip_safe=True,
    maintainer='francescopavesio',
    maintainer_email='francescopavesio@todo.todo',
    description='Neural Network Trajectory Node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_nn_node = robot_control_brain.trajectory_nn_node:main',
        ],
    },
)