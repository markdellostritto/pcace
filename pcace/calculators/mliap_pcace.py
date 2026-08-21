# the PCACE calculator for LAMMPS

from lammps.mliap.mliap_unified_abc import MLIAPUnified
import numpy as np
import torch
from ase.data import chemical_symbols

torch.set_default_dtype(torch.float64)

class MLIAP_PCACE(MLIAPUnified):
    def __init__(
        self, model, **kwargs
    ):
        super().__init__()
        # misc
        self.ndescriptors = 1 # ?
        self.nparams = 1 # ?
        self.device = "cpu"
        self.initialized = False
        self.force_cpu = False
        # cutoff
        self.rcutfac = 0.5 * float(model.rep.cutoff.rc) # Half of radial cutoff
        self.dtype = model.rep.cutoff.rc.dtype
        #print(f"rcutfac = {self.rcutfac}")
        #print(f"dtype = {self.dtype}")
        # elements
        self.element_types = [chemical_symbols[z] for z in model.rep.z_list]
        self.num_species = len(self.element_types)
        #print(f"element types = {self.element_types}")
        #print(f"num species = {self.num_species}")
        # model
        self.model = model
        for p in self.model.parameters(): p.requires_grad = False

    def _initialize_device(self, data):
        # kokkos
        using_kokkos = "kokkos" in data.__class__.__module__.lower()
        if using_kokkos and not self.force_cpu:
            device = torch.as_tensor(data.elems).device
        else:
            device = torch.device("cpu")
        self.device = device
        self.model = self.model.to(device)
        self.initialized = True

    def compute_forces(self, data):
        # atom data
        ntotal = data.ntotal # total number of atoms (local + ghost)
        nlocal = data.nlocal # number of atoms local to process
        nghost = ntotal - nlocal # number of ghost atoms
        npairs = data.npairs

        #print(f"N atoms total: {ntotal}")
        #print(f"N atoms local: {nlocal}")
        #print(f"N atoms ghost: {nghost}")
        #print(f"Atom indices: {data.iatoms}")
        #print(f"Atom types: {data.elems}")
        #print(f"Neighbor pairs: {npairs}")
        #print(f"Pair indices and displacement vectors: ")
        #print("\n".join([f"   ({i}, {j}), {r}" for i,j,r in zip(data.pair_i, data.pair_j, data.rij)]))

        # initialize device
        if not self.initialized: self._initialize_device(data)

        # if no owned atoms, do nothing
        if nlocal == 0 or npairs <= 1: return

        # make the batch
        z_array = [self.model.rep.z_list[int(e)] for e in data.elems]
        batch = {
            #"batch": torch.zeros(nlocal, dtype=torch.int64, device=self.device),
            "batch": torch.zeros(ntotal, dtype=torch.int64, device=self.device),
            "vectors": torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            "edge_index": torch.stack([
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],dim=0),
            "atomic_numbers": torch.tensor(z_array,dtype=torch.long).to(self.device),
            "positions": torch.zeros((nlocal,3)).to(self.dtype).to(self.device),
            "cell": torch.zeros((3,3)).to(self.dtype).to(self.device),
        }
        #print("batch = ",batch["batch"])
        #print("vectors = ",batch["vectors"])
        #print("edge_index = ",batch["edge_index"])
        #print("atomic_numbers = ",batch["atomic_numbers"])
        #print("positions = ",batch["positions"])
        #print("cell = ",batch["cell"])

        # compute energies and pair forces
        out = self.model(
            batch,
            training=False,
            compute_forces = False,
            compute_stress = False,
            compute_virials = False,
            compute_forces_edge = True,
        ) # compute_forces and compute_forces_edge can't both be true

        # compute the energy and forces
        node_energy = out["energy_sr_node"]
        #print("node_energy = ",node_energy)
        pair_forces = out["forces_edge_nnp"]
        #print("pair_forces = ",pair_forces)
        if pair_forces is None: pair_forces = torch.zeros_like(data["vectors"])
        if self.dtype == torch.float32: pair_forces = pair_forces.double()

        # update LAMMPS data
        eatoms = torch.as_tensor(data.eatoms)
        node_energy_real = node_energy[:nlocal].unsqueeze().detach()
        eatoms.copy_(node_energy_real)
        data.energy = node_energy_real.sum().item()
        data.update_pair_forces_gpu(pair_forces)
        
        #print(f"Energy: {node_energy_real.sum().item()}")
        #print(f"Pair indices and displacement vectors: ")
        #print("\n".join([f"   ({i}, {j}), {r}" for i,j,r in zip(data.pair_i, data.pair_j, data.rij)]))

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass