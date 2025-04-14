import os
from network_parameters import NetworkParameters


# ======================================================================================================================
#   Class NETWORK
# ======================================================================================================================
class Network:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.diagrams_dir = str()
        self.operational_data_file = str()
        self.data_loaded = False
        self.is_transmission = False
        self.baseMVA = 100.0
        self.nodes = list()
        self.loads = list()
        self.branches = list()
        self.generators = list()
        self.energy_storages = list()
        self.shared_energy_storages = list()
        self.prob_operation_scenarios = list()          # Probability of operation (generation and consumption) scenarios
        self.cost_energy_p = list()
        self.cost_flex = list()
        self.active_distribution_network_nodes = list()
        self.params = NetworkParameters()


    def read_network_parameters(self, params_filename):
        filename = os.path.join(self.data_dir, self.name, params_filename)
        self.params.read_parameters_from_file(filename)