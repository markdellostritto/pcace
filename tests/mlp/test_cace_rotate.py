import torch
import numpy as np
import ase
from ase.io import read, write
from pcace.data import Molecule
from pcace.mlp import Cace
from pcace.basis import CutoffCos
from pcace.basis import RadialBessel
from pcace.basis import AngularBasis
from pcace import torch_geometric

#********************************************************
# Files
#********************************************************

print("==========================================================================")
filename="h2o_pbcf.xyz"
data_key = {'energy': 'potential_energy', 'forces':'forces'}
#filename="Ar_cluster_n3.xyz"
#data_key = {'energy': 'energy', 'forces':'forces'}
print("filename = ",filename)
print("data_key = ",data_key)

#********************************************************
# Read Coordinates
#********************************************************

print("==========================================================================")
print("Reading Coordinates")
struc = ase.io.read(filename, index = 0, format = "extxyz") 
#struc = ase.io.read(filename, index = 0, format = "extxyz") 
a1=0.0
a2=0.0
a3=0.0
#a1=13.0
#a2=28.0
#a3=78.0
angle1=a1*torch.pi/180.0
angle2=a2*torch.pi/180.0
angle3=a3*torch.pi/180.0
rmat1=torch.tensor(
    [
        [np.cos(angle1),-np.sin(angle1),0.0],
        [np.sin(angle1),np.cos(angle1),0.0],
        [0.0,0.0,1.0]
    ],dtype=torch.float
)
rmat2=torch.tensor(
    [
        [np.cos(angle2),0.0,np.sin(angle2)],
        [0.0,1.0,0.0],
        [-np.sin(angle2),0.0,np.cos(angle2)]
    ],dtype=torch.float
)
rmat3=torch.tensor(
    [
        [1.0,0.0,0.0],
        [0.0,np.cos(angle3),-np.sin(angle3)],
        [0.0,np.sin(angle3),np.cos(angle3)]
    ],dtype=torch.float
)
print("angle1 = ",angle1)
print("angle2 = ",angle2)
print("angle3 = ",angle3)
print("rmat1 = ",rmat1)
print("rmat2 = ",rmat2)
print("rmat3 = ",rmat3)
print(struc)
print(struc.todict())

#********************************************************
# Make Molecule
#********************************************************

print("==========================================================================")
print("Making Molecule")
rc=6.0
mol=Molecule.from_atoms(
    struc,
    cutoff=rc,
    data_key=data_key
)
mol.batch=1
# read data
print(mol)
print("batch = ",mol.batch)
print("cell = \n",mol.cell)
print("posns = \n",mol.positions)
print("shifts = \n",mol.shifts)
print("unit_shifts = \n",mol.unit_shifts)
print("energy = \n",mol.energy)
print("forces = \n",mol.forces)
print("edge_index = \n",mol.edge_index)
print("rotated = ",
    torch.matmul(torch.matmul(torch.matmul(mol.positions,rmat1),rmat2),rmat3)
)
struc.positions=torch.matmul(torch.matmul(torch.matmul(mol.positions,rmat1),rmat2),rmat3).detach().numpy()
print(struc.todict())

#********************************************************
# Make CACE
#********************************************************

print("==========================================================================")
print("Making CACE")
cace = Cace(
    z_list = [1,8],
    order = 3,
    cutoff = CutoffCos(rc),
    radial = RadialBessel(rc=rc,nr=8),
    angular = AngularBasis(l_max=3),
    dim_node_embed = 2,
    dim_radial_embed = 4,
    calc_stress = False,
    calc_virials = False,
)
#print("z_list = ",cace.z_list)
#print("order = ",cace.order)
#print("cutoff = ",cace.cutoff)
#print("radial = ",cace.radial)
#print("angular = ",cace.angular)
#print("dim_node_embed = ",cace.dim_node_embed)
#print("dim_radial_embed = ",cace.dim_radial_embed)
#print("node_encoder = ",cace.node_encoder)
#print("node_embedder_send = ",cace.node_embedder_send)
#print("node_embedder_recv = ",cace.node_embedder_recv)
#print("edge_encoder = ",cace.edge_encoder)
#print("dim_edge_encode = ",cace.dim_edge_encode)
print(cace)

#********************************************************
# Compute CACE
#********************************************************

print("==========================================================================")
print("Computing CACE")
#cace.forward(mol.to_dict())
data_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
        Molecule.from_atoms(
            struc, cutoff=rc
        )
    ],
    batch_size=1,
    shuffle=False,
    drop_last=False,
)
batch_base = next(iter(data_loader))
batch = batch_base.clone()
output = cace(batch.to_dict(), training=True)
print(output['energy'])
print(output['forces'])

angle1=-a1*torch.pi/180.0
angle2=-a2*torch.pi/180.0
angle3=-a3*torch.pi/180.0
rmat1=torch.tensor(
    [
        [np.cos(angle1),-np.sin(angle1),0.0],
        [np.sin(angle1),np.cos(angle1),0.0],
        [0.0,0.0,1.0]
    ],dtype=torch.float
)
rmat2=torch.tensor(
    [
        [np.cos(angle2),0.0,np.sin(angle2)],
        [0.0,1.0,0.0],
        [-np.sin(angle2),0.0,np.cos(angle2)]
    ],dtype=torch.float
)
rmat3=torch.tensor(
    [
        [1.0,0.0,0.0],
        [0.0,np.cos(angle3),-np.sin(angle3)],
        [0.0,np.sin(angle3),np.cos(angle3)]
    ],dtype=torch.float
)
print(torch.matmul(torch.matmul(torch.matmul(output['forces'],rmat3),rmat2),rmat1))
#energy_output = to_numpy(output[self.energy_key])
#forces_output = to_numpy(output[self.forces_key])

