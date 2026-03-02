import ase
from ase.io import read, write
from ase.neighborlist import NeighborList

ifile="h2o_pbcf.xyz"
atoms=ase.io.read(ifile,format="xyz")
cutoff=1.1

print("Atoms = ")
print(atoms)
print(atoms.numbers)
print(atoms.positions)

i,j,d,s = ase.neighborlist.primitive_neighbor_list(
        quantities = "ijdS",
        pbc = atoms.pbc,
        positions = atoms.positions,
        cutoff = cutoff,
        self_interaction = False,
        use_scaled_positions = False,
        cell = atoms.cell
    )
print("neighbor list - beg node = ",i)
print("neighbor list - end node = ",j)
print("neighbor list - dist     = ",d)
print("neighbor list - shifts   = ",s)

#nl = ase.ase.neighborlist.NeighborList(
#        cutoffs=cutoff,
#        self_interaction=True
#    )
#nl.update(atoms)
#print(nl)

