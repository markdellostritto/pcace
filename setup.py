from setuptools import setup, find_packages

setup(
    name='PCACE',
    version='0.1.0',
    author='Mark DelloStritto',
    author_email='mark.dellostritto@temple.edu',
    description='Cartesian Multilayer Atomic Cluster Expansion Machine Learning Potential',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'ase',
        'torch',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

