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
    compute_forces: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    # compute total force/stress by taking gradient w.r.t. atomic position
    if (compute_virials or compute_stress) and displacement is not None:
        #print("computing stress and virials")
        forces, virials, stress = get_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_forces:
        #print("computing force only")
        forces, virials, stress = (
            get_forces(
                energy=energy, 
                positions=positions, 
                training=training
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    # return
    return forces, virials, stress

"""
    get_forces
    Compute the forces from the inputs using autograd.
    Arguments:
    energy - The energy yielded by a given ANN.  This may be computed directly from
        the output of the NN, or it may be transformed from the output of the network, 
        i.e. if the network yields an atomic charge or coefficient which is used in a
        function to compute the energy.
    positions - The inputs to the ANN used to compute the energy.  This can be the atomic
        positions, however, this can also be the edge vectors between the atoms.  In these
        cases different forces are produced.  If the inputs are positions than the ouput is
        the total force on each atom.  If the inputs are the edge vectors than the ouput is
        a list of all forces between atoms (the "edge forces").
    training - Whether the ANN is bein trained, thereby ensuring that the graph is not 
        destroyed and that a graph for the second derivative is created during training.
"""
def get_forces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    training: bool = False
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # allows gradients for non-mathematical connections
    )[0]  # [n_nodes, 3]
    if gradient is None: return torch.zeros_like(positions)
    else: return -1 * gradient

"""
    get_forces_virials
    Compute the forces, virials, and stress of a given system.
    Argument:
    energy - The energy yielded by a given ANN.  This may be computed directly from
        the output of the NN, or it may be transformed from the output of the network, 
        i.e. if the network yields an atomic charge or coefficient which is used in a
        function to compute the energy.
    positions - The inputs to the ANN used to compute the energy, importantly assuming
        that the inputs are in fact the atomic positions.  Unlike "get_forces", here we must
        assume that the inputs are positions, otherwise the calculation of the stress does
        not make mathematical sense.
    displacements - ?
    cell - The unit cell matrices.
    training - Whether the ANN is bein trained, thereby ensuring that the graph is not 
            destroyed and that a graph for the second derivative is created during training.
    compute_stress - Whether the total stress is computed along with the forces and virials.
"""
def get_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = False,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # check the dimension of the energy tensor
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True, # allows gradients for non-mathematical connections
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

