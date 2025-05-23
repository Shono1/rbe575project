from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rbe575project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jhkeselman',
    maintainer_email='jhkeselman@wpi.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_controller = rbe575project.safety_controller:main',
            'cbf_node = rbe575project.lib.projectcode.cbfnode:main'
        ],
    },
)
