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
        energy_key: str, key for energy in model output
        forces_key: str, key for forces in model output
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        atomic_energies: dict, dictionary of atomic energies to add to model output
    """
    def __init__(
        self,
        model_path: Union[str, torch.nn.Module],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        compute_stress = False,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        charge_key: str = None,
        charge_unit: float = 1.0/(90.0474)**0.5, # the standard normal factor in accordance with the cace convention used in ewald.py
        data_key: dict = None,
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

        # load the model
        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, map_location=device)
        elif isinstance(model_path, torch.nn.Module):
            self.model = model_path
        else:
            raise ValueError("model_path must be a string or nn.Module")
        self.model.to(device)

        # initialize the device
        self.device = tools.init_device(device)

        # set the units
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.charge_unit = charge_unit

        # set the cutoff 
        self.cutoff = self.model.cutoff.rc.clone().detach().item()
        
        # set atomic energies        
        self.atomic_energies = atomic_energies

        # set data keys
        #print("setting data keys")
        self.compute_stress = compute_stress
        self.model.calc_stress = compute_stress
        #print("calc_stress = ",self.model.calc_stress)
        self.energy_key = energy_key 
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.charge_key = charge_key
        self.data_key = data_key

        # turn off gradients for efficiency
        for param in self.model.parameters():
            param.requires_grad = False

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        #print("calculate")
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)
        
        # prepare data - make a dataset with only one structure
        #print("preparing data")
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                Molecule.from_atoms(
                    atoms, cutoff=self.cutoff,
                    data_key=self.data_key,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # get the structure in the batch
        #print("getting the next batch")
        batch = next(iter(data_loader)).to(self.device).clone()

        # compute energy, force, stress
        #print("computing energy, force, stress")
        output = self.model(batch.to_dict(), training=True)
        energy_output = output[self.energy_key].cpu().detach().numpy()
        forces_output = output[self.forces_key].cpu().detach().numpy()

        # subtract atomic energies if available
        #print("subtracting atomic energies")
        if self.atomic_energies:
            e0 = sum(self.atomic_energies.get(Z, 0) for Z in atoms.get_atomic_numbers())
        else:
            e0 = 0.0
        
        # set energy, force, stress
        #print("setting energy, force, stress")
        self.results["energy"] = (energy_output + e0) * self.energy_units_to_eV
        self.results["forces"] = forces_output * self.energy_units_to_eV / self.length_units_to_A
        #print("compute_stress = ",self.compute_stress)
        #print("stress_key = ",self.stress_key)
        #print("output = ",output)
        if self.compute_stress and output[self.stress_key] is not None:
            stress = output[self.stress_key].cpu().detach().numpy()
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])

        # return results
        #print("return")
        return self.results