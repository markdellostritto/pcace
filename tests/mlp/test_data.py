# pytorch 
import torch
# data
from pcace.data.load_data import get_dataset_from_xyz
from pcace.data.load_data import load_data_loader
# mlp
from pcace.mlp import Cace
# basis
from pcace.basis import CutoffCos
from pcace.basis import RadialBessel
from pcace.basis import AngularBasis

#********************************************************
# Read XYZ Files
#********************************************************

print("================================================")
print("Reading Coordinates")
filename="water_n2pt.xyz"
subset = get_dataset_from_xyz(
    train_path = filename,
    cutoff = 6.0,
    valid_path = None,
    valid_fraction = 0.1,
    test_path = None,
    seed = 1,
    data_key={'energy': 'potential_energy', 'forces':'forces'},
    atomic_energies={1: -13.605693122994, 8: -422.507792818175},
)
print("train = ",subset.train)
print("valid = ",subset.valid)

print("================================================")
print("Making Data Loaders")

train_loader = load_data_loader(
    collection = subset,
    data_type = "train",
    batch_size = 2,
)
valid_loader = load_data_loader(
    collection = subset,
    data_type = "valid",
    batch_size = 2,
)
#print("train_loader = ",train_loader)
#print("valid_loader = ",valid_loader)

print("================================================")
print("Making CACE")
rc=6.0
cace = Cace(
    z_list = [1,8],
    order = 2,
    cutoff = CutoffCos(rc),
    radial = RadialBessel(rc=rc,nr=8),
    angular = AngularBasis(l_max=3),
    dim_node_embed = 2,
    dim_radial_embed = 4,
    n_hidden = [16,16]
)
print("z_list = ",cace.z_list)
print("order = ",cace.order)
print("cutoff = ",cace.cutoff)
print("radial = ",cace.radial)
print("angular = ",cace.angular)
print("dim_node_embed = ",cace.dim_node_embed)
print("dim_radial_embed = ",cace.dim_radial_embed)
print("node_encoder = ",cace.node_encoder)
print("node_embedder_send = ",cace.node_embedder_send)
print("node_embedder_recv = ",cace.node_embedder_recv)
print("edge_encoder = ",cace.edge_encoder)
print("dim_edge_encode = ",cace.dim_edge_encode)

print("================================================")
print("Computing CACE")

for batch in train_loader:
    print("batch = ",batch)
    print(batch.batch)
    unique, counts = torch.unique(batch.batch,return_counts=True)
    print("unique = ",unique)
    print("counts = ",counts)
    print("test = ",torch.tensor([counts[b] for b in batch.batch]))
    outputs = cace.forward(batch.to_dict())
    print("energy = ",outputs["energy"])
    print("energy/counts = ",outputs["energy"]/counts)
    print("forces = ",outputs["forces"])
