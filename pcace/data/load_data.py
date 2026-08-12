import dataclasses
import logging
from typing import Dict, List, Tuple, Sequence
import numpy as np
import ase
import ase.io
from ase import Atoms
from .. import torch_geometric
from ..data import Molecule, key_data_default

__all__ = [
    "load_data_loader",
    "read_atoms_xyz",
    "read_dataset_xyz",
    "random_train_valid_split",
]

@dataclasses.dataclass
class SubsetAtoms:
    train: Atoms
    valid: Atoms 
    test:  Atoms
    cutoff: float
    key_data: Dict
    atomic_energies: Dict 

def load_data_loader(
    collection: SubsetAtoms,
    data_type: str, # ['train', 'valid', 'test']
    batch_size: int,
):
    #print("load_data_loader")
    # ==== check the types ====
    allowed_types = ['train', 'valid', 'test']
    if data_type not in allowed_types: raise ValueError(f"Input value must be one of {allowed_types}, got {data_type}")
    # ==== set the cutoff, key, and energies ====
    cutoff = collection.cutoff
    key_data = collection.key_data
    atomic_energies = collection.atomic_energies
    # ==== make the data loader ====
    if data_type == 'train':
        loader = torch_geometric.DataLoader(
            dataset=[
                Molecule.from_atoms(atoms, cutoff=cutoff, key_data=key_data, atomic_energies=atomic_energies)
                for atoms in collection.train
            ],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    elif data_type == 'valid':
        loader = torch_geometric.DataLoader(
            dataset=[
                Molecule.from_atoms(atoms, cutoff=cutoff, key_data=key_data, atomic_energies=atomic_energies)
                for atoms in collection.valid
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    elif data_type == 'test':
        loader = torch_geometric.DataLoader(
            dataset=[
                Molecule.from_atoms(atoms, cutoff=cutoff, key_data=key_data, atomic_energies=atomic_energies)
                for atoms in collection.test
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    # ==== return the data loader ====
    return loader

"""
    Read a list of atoms from an xyz file using ASE.
    The energy, force, and stress are properly read in for any
    version of ASE using atomis.info and atoms.arrays.
"""
def read_atoms_xyz(
    path_file: str,
    key_data: Dict[str, str] = None,    
) -> Atoms:
    #print("read_atoms_xyz")
    if key_data is not None:
        key_data = key_data_default | key_data
    else:
        key_data = key_data_default
    # ==== read the atoms ====
    atoms = ase.io.read(path_file, index=":")
    # ==== store the energy ====
    if key_data["energy"] == "energy": 
        logging.warning("key_energy 'energy' is no longer safe.")
        key_data["energy"] = key_data_default["energy"]
        for atom in atoms:
            try:
                atom.info[key_data["energy"]] = atom.get_potential_energy()
            except Exception as e:  
                #logging.error(f"Failed to extract energy: {e}")
                atom.info[key_data["energy"]] = None    
    # ==== store the force ====
    if key_data["forces"] == "forces": 
        logging.warning("key_force 'forces' is no longer safe.")
        key_data["forces"] = key_data_default["forces"]
        for atom in atoms:
            try:
                atom.arrays[key_data["forces"]] = atom.get_forces()
            except Exception as e:  
                #logging.error(f"Failed to extract forces: {e}")
                atom.arrays[key_data["forces"]] = None
    # ==== store the stress ====
    if key_data["stress"] == "stress": 
        logging.warning("key_stress 'stress' is no longer safe.")
        key_data["stress"] = key_data_default["stress"]
        for atom in atoms:
            try:
                atom.info[key_data["stress"]] = atom.get_stress()
            except Exception as e:  
                #logging.error(f"Failed to extract stress: {e}")
                atom.info[key_data["stress"]] = None
    # ==== return the atoms ====
    return atoms, key_data

"""
    Load training and test dataset from xyz file
"""
def read_dataset_xyz(
    cutoff: float,
    path_train: str,
    path_valid: str = None,
    path_test: str = None,
    valid_fraction: float = 0.1,
    seed: int = 1234,
    key_data: Dict[str, str] = None,
    atomic_energies: Dict[int, float] = None
) -> SubsetAtoms:
    #print("read_dataset_xyz")
    # ==== read atoms ====
    atoms, key_data = read_atoms_xyz(
        path_file = path_train,
        key_data = key_data
    )
    if not isinstance(atoms, list): atoms = [atoms]
    logging.info(
        f"Loaded {len(atoms)} training configurations from '{path_train}'"
    )
    # ==== load train/valid ====
    if path_valid is not None:
        atoms_valid = ase.io.read(path_valid, index=":")
        if not isinstance(atoms_valid, list): atoms_valid = [atoms_valid]
        logging.info(
            f"Loaded {len(atoms_valid)} validation configurations from '{path_valid}'"
        )
        atoms_train = atoms
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        atoms_train, atoms_valid = random_train_valid_split(
            atoms, valid_fraction, seed
        )
    # ==== load test ====
    atoms_test = []
    if path_test is not None:
        atoms_test = ase.io.read(path_test, index=":")
        if not isinstance(atoms_test, list):
            atoms_test = [atoms_test]
        logging.info(
            f"Loaded {len(atoms_test)} test configurations from '{path_test}'"
        )
    # ==== return subset ====
    return (
        SubsetAtoms(
            train = atoms_train, 
            valid = atoms_valid, 
            test  = atoms_test, 
            cutoff = cutoff, 
            key_data = key_data, 
            atomic_energies = atomic_energies
        )
    )

def random_train_valid_split(
    items: Sequence,
    valid_fraction: float,
    seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0
    size = len(items)
    train_size = size - int(valid_fraction * size)
    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )
