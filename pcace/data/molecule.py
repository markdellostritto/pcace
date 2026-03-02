import numpy as np
import torch
from ase import Atoms
from typing import Optional, Dict, Sequence
from .. import torch_geometric
from .neighborhood import get_neighborhood

default_data_key = {
    "energy": "energy",
    "forces": "forces",
    "stress": "stress",
    "virials": "virials",
}

class Molecule(torch_geometric.data.Data):
    # batch
    num_graphs: torch.Tensor
    batch: torch.Tensor
    # graph
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    # global
    energy: torch.Tensor
    # cell
    cell: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    # atom data
    atomic_numbers: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    forces: torch.Tensor
    
    # ==== initialization ====
    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        atomic_numbers: torch.Tensor,  # [n_nodes]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        num_nodes: Optional[torch.Tensor] = None, #[,]
        cell: Optional[torch.Tensor] = None,  # [3,3]
        forces: Optional[torch.Tensor] = None,  # [n_nodes, 3]
        energy: Optional[torch.Tensor] = None,  # [, ]
        stress: Optional[torch.Tensor] = None,  # [1,3,3]
        virials: Optional[torch.Tensor] = None,  # [1,3,3]
        additional_info: Optional[Dict] = None, 
    ):
        # Check shapes
        if num_nodes is None: num_nodes = atomic_numbers.shape[0]
        else: assert num_nodes == atomic_numbers.shape[0]
        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        
        # Aggregate data
        data = {
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "atomic_numbers": atomic_numbers,
            "num_nodes": num_nodes,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
        }
        if additional_info is not None:
            data.update(additional_info)
        super().__init__(**data)

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms, 
        cutoff: float,
        data_key: Dict[str, str] = None,
        atomic_energies: Optional[Dict[int, float]] = None,
    ) -> "Molecule":
        if data_key is not None:
            data_key = default_data_key.update(data_key)
        data_key = default_data_key
        
        positions = atoms.get_positions()
        pbc = tuple(atoms.get_pbc())
        cell = np.array(atoms.get_cell())
        atomic_numbers = atoms.get_atomic_numbers()

        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell
        )

        # get total energy
        energy = atoms.info.get(data_key["energy"], None)  # eV
        if energy is None and data_key['energy'] == 'energy':
            try:
                energy = atoms.get_potential_energy()
            except:
                energy = None
        # subtract atomic energies if available
        if atomic_energies and energy is not None:
            energy -= sum(atomic_energies.get(Z, 0) for Z in atomic_numbers)
        
        try:
            forces = atoms.arrays.get(data_key["forces"], None)  # eV / Ang
        except:
            if data_key['forces'] == 'forces': forces = atoms.get_forces()
            else: forces = None
        stress = atoms.info.get(data_key["stress"], None)  # eV / Ang
        virials = atoms.info.get(data_key["virials"], None)
        
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )
        forces = (
            torch.tensor(forces, dtype=torch.get_default_dtype())
            if forces is not None
            else None
        )
        energy = (
            torch.tensor(energy, dtype=torch.get_default_dtype())
            if energy is not None
            else None
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if stress is not None
            else None
        )
        virials = (
            torch.tensor(virials, dtype=torch.get_default_dtype()).unsqueeze(0)
            if virials is not None
            else None
        )

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
        )
    
def get_data_loader(
    dataset: Sequence[Molecule],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

"""
    Convert voigt notation to matrix notation
    :param t: (6,) tensor or (3, 3) tensor
    :return: (3, 3) tensor
"""
def voigt_to_matrix(t: torch.Tensor):
    if t.shape == (3, 3): return t
    return torch.tensor(
        [[t[0], t[5], t[4]], [t[5], t[1], t[3]], [t[4], t[3], t[2]]], dtype=t.dtype
    )
