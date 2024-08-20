from typing import List
from .module import Module
from forte.data import ForteData
from forte._forte import RDMsType


class ActiveSpacePDMs(Module):

    """
    A module to prepare an ActiveSpaceIntegral
    """

    def __init__(self, max_rdm_level: int, rdms_type=RDMsType.spin_dependent):
        """
        Parameters
        ----------
        max_rdm_level: int
            The maximum level of RDMs to be computed.
        """
        super().__init__()
        self.max_rdm_level = max_rdm_level
        self.rdms_type = rdms_type
    # def _run(self, data: ForteData):
    #     import forte
    #     all_pdms = data.active_space_solver.compute_pdms(data.as_ints, self.max_rdm_level)
    #     print('--------------test---pdms--------------')
    #     ue = data.active_space_solver.compute_contracted_energy(data.as_ints, self.max_rdm_level)
    #     print('--------------test---contracted_energy--------------')
    #     print(ue)
    #     return all_pdms
    def _run(self, data: ForteData):
        import forte
        hamiltonian, substates = data.active_space_solver.get_hamiltonian(data.as_ints, self.max_rdm_level)
        return hamiltonian, substates
