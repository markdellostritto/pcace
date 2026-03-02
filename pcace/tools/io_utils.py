import logging
import os
import sys
import numpy as np
from typing import Optional, Union, Dict, List
from ase import Atoms

def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

"""
    Read a multi-frame XYZ file and return a list of unique atomic numbers
    present across all frames.
    Returns: list: List of unique atomic numbers.
"""    
def get_unique_atomic_number(atoms_list: List[Atoms]) -> List[int]:
    unique_atomic_numbers = set()
    for atoms in atoms_list:
        unique_atomic_numbers.update(atom.number for atom in atoms)
    return list(unique_atomic_numbers)

"""
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
"""
def compute_average_E0s(
    atom_list: Atoms, zs: List[int] = None, energy_key: str = "energy"
) -> Dict[int, float]:
    len_xyz = len(atom_list)
    if zs is None:
        zs = get_unique_atomic_number(atom_list)
        zs.sort() # sort by atomic number
    len_zs = len(zs)

    A = np.zeros((len_xyz, len_zs))
    B = np.zeros(len_xyz)
    for i in range(len_xyz):
        B[i] = atom_list[i].info[energy_key]
        #B[i] = atom_list[i].get_potential_energy()
        for j, z in enumerate(zs):
            A[i, j] = np.count_nonzero(atom_list[i].get_atomic_numbers() == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict
