from setuptools import setup

setup(
    name='GridWorld',
    version='1.0.0',
    description='GridWorld',
    url='None',
    author='Ercument Ilhan',
    author_email='e.ilhan@qmul.ac.uk',
    license='GPL',
    packages=['gridworld'],
    install_requires=[
        'gym>=0.17.1',
        'numpy>=1.18.1',
        'opencv-python>=4.2.0.34',
        'pathfinding>=0.0.4'
    ])
