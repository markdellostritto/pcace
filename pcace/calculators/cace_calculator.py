# the PCACE calculator for ASE

from typing import Union
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .. import torch_geometric, tools
from ..data import Molecule
 
__all__ = ["CACECalculator"]

class CACECalculator(Calculator):
    """CACE ASE Calculator
    args:
        model_path: str or nn.module, path to model
        device: str, device to run on (cuda or cpu)
        compute_stress: bool, whether to compute stress
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        atomic_energies: dict, dictionary of atomic energies to add to model output
    """
    # ==== initialization ====
    def __init__(
        self,
        model_path: Union[str, torch.nn.Module],
        device: str,
        # units
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        charge_unit: float = 1.0/(90.0474)**0.5, # the standard normal factor in accordance with the cace convention used in ewald.py
        # flags - calculation
        compute_stress = False,
        # keys - calculation
        key_data: dict  = None,
        # energies
        atomic_energies: dict = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]
        self.results = {}

        # == load the model ==
        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, map_location=device)
        elif isinstance(model_path, torch.nn.Module):
            self.model = model_path
        else:
            raise ValueError("model_path must be a string or nn.Module")
        self.model.to(device)

        # == initialize the device ==
        self.device = tools.init_device(device)

        # == set the units ==
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.charge_unit = charge_unit

        # == set the cutoff ==
        self.cutoff = self.model.rep.cutoff.rc.clone().detach().item()
        
        # == set atomic energies ==
        self.atomic_energies = atomic_energies

        # == set data keys ==
        #print("setting data keys")
        self.compute_stress = compute_stress
        self.model.compute_stress = compute_stress
        self.key_data   = key_data

        # turn off gradients for efficiency
        for param in self.model.parameters():
            param.requires_grad = False

    # ==== calculation ====
    def calculate(self, 
        atoms=None,
        properties=None, 
        system_changes=all_changes
    ):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        #print("calculate")
        # == call to base-class to set atoms attribute ==
        Calculator.calculate(self, atoms)
        
        # == prepare data - make a dataset with only one structure ==
        #print("preparing data")
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                Molecule.from_atoms(
                    atoms,
                    cutoff=self.cutoff,
                    key_data=self.key_data,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # == get the structure in the batch ==
        #print("getting the next batch")
        batch = next(iter(data_loader)).to(self.device).clone()

        # == compute energy, force, stress ==
        #print("computing energy, force, stress")
        output = self.model(
            batch.to_dict(),
            training=True
        )
        energy_output = output[self.model.key_energy].cpu().detach().numpy()[0]
        forces_output = output[self.model.key_forces].cpu().detach().numpy()

        # == subtract atomic energies if available ==
        #print("subtracting atomic energies")
        if self.atomic_energies:
            e0 = sum(self.atomic_energies.get(Z, 0) for Z in atoms.get_atomic_numbers())
        else:
            e0 = 0.0
        
        # == set energy, force, stress ==
        #print("setting energy, force, stress")
        self.results["energy"] = (energy_output + e0) * self.energy_units_to_eV
        self.results["forces"] = forces_output * self.energy_units_to_eV / self.length_units_to_A
        if self.compute_stress and output[self.model.key_stress] is not None:
            stress = output[self.model.key_stress].cpu().detach().numpy()
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])

        # == return results ==
        return self.results