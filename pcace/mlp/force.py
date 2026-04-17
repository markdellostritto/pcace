#****************************************************
# Import Statements
#****************************************************

import torch
from typing import Optional, List, Tuple

#****************************************************
# Force Functions
#****************************************************

def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (compute_virials or compute_stress) and displacement is not None:
        #print("computing stress and virials")
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force:
        #print("computing force only")
        forces, virials, stress = (
            compute_forces(energy=energy, positions=positions, training=training),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    return forces, virials, stress

def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = False
) -> torch.Tensor:
    # check the dimension of the energy tensor
    if len(energy.shape) == 1:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        gradient = torch.autograd.grad(
            outputs=[energy],  # [n_graphs, ]
            inputs=[positions],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=training,  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]  # [n_nodes, 3]
    else:
        num_energy = energy.shape[1]
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy[:,0])]
        gradient = torch.stack([ 
            torch.autograd.grad(
                outputs=[energy[:,i]],  # [n_graphs, ]
                inputs=[positions],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or (i < num_energy - 1)),  # Make sure the graph is not destroyed during training
                create_graph=(training or (i < num_energy - 1)),  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
                )[0] for i in range(num_energy) 
           ], axis=2)  # [n_nodes, 3, num_energy]

    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = False,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # check the dimension of the energy tensor
    if len(energy.shape) == 1:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        gradient, virials = torch.autograd.grad(
            outputs=[energy],  # [n_graphs, ]
            inputs=[positions, displacement],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=training,  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,
        )
        stress = torch.zeros_like(displacement)
        if compute_stress and virials is not None:
            cell = cell.view(-1, 3, 3)
            volume = torch.einsum(
                "zi,zi->z",
                cell[:, 0, :],
                torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
            ).unsqueeze(-1)
            stress = virials / volume.view(-1, 1, 1)
    else:
        num_energy = energy.shape[1]
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy[:,0])]
        gradient_list, virials_list, stress_list = [], [], []
        for i in range(num_energy):
            gradient, virials = torch.autograd.grad(
                outputs=[energy[:,i]],  # [n_graphs, ]
                inputs=[positions, displacement],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or (i < num_energy - 1)),  # Make sure the graph is not destroyed during training
                create_graph=(training or (i < num_energy - 1)),  # Create graph for second derivative
                allow_unused=True,
            )
            stress = torch.zeros_like(displacement)
            if compute_stress and virials is not None:
                cell = cell.view(-1, 3, 3)
                volume = torch.einsum(
                    "zi,zi->z",
                    cell[:, 0, :],
                    torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                ).unsqueeze(-1)
                stress = virials / volume.view(-1, 1, 1)
            gradient_list.append(gradient)
            virials_list.append(virials)
            stress_list.append(stress)
        gradient = torch.stack(gradient_list, axis=2)
        virials = torch.stack(virials_list, axis=-1)
        stress = torch.stack(stress_list, axis=-1)
    # return zero in case of an error
    if gradient is None: gradient = torch.zeros_like(positions)
    if virials is None: virials = torch.zeros((1, 3, 3))
    # multiply by -1 
    return -1 * gradient, -1 * virials, stress

def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # create zero cell if none exists
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3, 3,
            dtype=positions.dtype,
            device=positions.device,
        )
    # make the tensor to store the displacement
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    # symmetrize the displacement
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    sender = edge_index[0]
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement, cell

