# PCACE 

Author: Mark J. DelloStritto

A code to train neural network potentials using atomic structures and associated references energies and forces.
This code is based on the Cartesian Atomic Cluster Expansion methodology[1,2], where the local density around each 
atom is expanded in terms of products of radial functions and Cartesian Gaussians.  The atomic properties
are then fit to powers of this expansion, with the power of the expansion determining the body-order of the 
representation of the local density.

## CODE ORGANIZATION

**BASIS**
* cutoff.py	- Cutoff functions for basis
* radial.py	- Radial basis functions
* angular.py 	- Angular basis functions
* product.py	- Classes for generating and storing products of basis functions

**CALCULATORS**
* cace_calulator.py	- CACE Calculator for the ASE library[3]

**DATA**
* molecule.py		- Class storing atomic structures and associated properties
* neighborhood.py	- Class for computing atomic neighborhoods up to a given cutoff radius
* load_data.py		- Functions for loading atomic data into training/validation sets

**ML**
* afunc.py	- Custom neural network activation functions
* blocks.py	- Neural network blocks
* loss_fn.py	- Loss functions
* loss_map.py	- Mapping of different targets to loss functions
* metric.py	- Metrics for measuring neural network performance
* yogi.py	- YOGI optimizer

**MLP**
* cace.py		- Cartesian Atomic Cluster Expansion local coordinated representation
* type.py		- Encodings/Embeddings of Elements/Types
* force.py		- Class to compute forces from atomic structures and energies
* nnp.py		- Neural Network Potential - Local Rep. + Atomic Neural Networks
* ann_sr.py		- ANN - short-range energies
* ann_pauli_gauss.py	- ANN - Z - Pauli repulsion - Gaussian overlap
* ann_pauli_sech.py	- ANN - Z - Pauli repulsion - Logistic overlap
* ann_london_cut.py	- ANN - C6 - London dispersion - Cutoff
* ann_london_long.py	- ANN - C6 - London dispersion - Ewald
* ann_ldamp_cut.py	- ANN - C6 - Damped dispersion - Cutoff
* ann_ldamp_long.py	- ANN - C6 - Damped dispersion - Ewald
* ann_coul_long.py	- ANN - Q - Coulomb interaction - Ewald
* ann_grho_long.py	- ANN - Q - Gaussian charge density interaction - Ewald

**OPT**
* train.py	- Functions for training a NNP

**TOOLS**
* device.py	- Tools for setting the device
* io_utils.py	- Tools for input/output
* scatter.py	- Tools for selective broadcasts/reductions over tensors

**TORCH_GEOMETRIC**
* data.py	- Class for storing a single graph
* dataset.py	- Class for contiguously storing multiple graphs
* batch.py	- Class for creating a batch of graphs
* dataloader.py	- Class for loading graph data
* utils.py	- Tools for storing/extracting graph data

## INSTALLATION

Installation in a given environment can be acheived one of two ways.
If using an older version of setuptools one can directly install the code by accessing 
setup.py from the command line:

python setup.py build

python setup.py install

Modern versions of setuptools have deprecated setup.py as a command line tool.  Thus, 
one should instead use pip with the following command in the same folder as setup.py:

pip3 install .
	
## TRAINING - DATA

All training data is stored contiguously in one extended xyz file

## REFERENCES

[1] R. Drautz, “Atomic cluster expansion of scalar, vectorial, and tensorial properties including magnetism and charge transfer,” Phys. Rev. B 102(2), 024104 (2020)
[2] B. Cheng, “Cartesian atomic cluster expansion for machine learning interatomic potentials,” Npj Comput Mater 10(1), 1–10 (2024)
[3] https://ase-lib.org/
