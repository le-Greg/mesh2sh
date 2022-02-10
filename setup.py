from setuptools import setup, find_packages

setup(
    name='mesh2sh',
    version='1.0',
    url='https://github.com/le-Greg/mesh2sh.git',
    author='le-Greg',
    packages=find_packages(include=['mesh2sh', 'mesh2sh.*']),
    install_requires=[
        'torch >= 1.8',
        'scipy >= 1.7',
        'scikit-sparse',
        'pyshtools'],
)
