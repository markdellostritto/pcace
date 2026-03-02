from setuptools import setup, find_packages

setup(
    name='CMACE',
    version='0.1.0',
    author='Mark DelloStritto',
    author_email='mark.dellostritto@temple.edu',
    description='Cartesian Multilayer Atomic Cluster Expansion Machine Learning Potential',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'ase==3.22.1',
        'torch==2.6.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

