import os
from node import Node
from load import Load
from branch import Branch
from generator import Generator
from energy_storage import EnergyStorage
from network_parameters import NetworkParameters
from helper_functions import *


# ======================================================================================================================
#   Class NETWORK
# ======================================================================================================================
class Network:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.diagrams_dir = str()
        self.num_instants = 24
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

    def read_network_data(self, operational_data_filename):
        _read_network_data(self, operational_data_filename)

    def read_network_from_json_file(self, network_filename):
        filename = os.path.join(self.data_dir, self.name, network_filename)
        _read_network_from_json_file(self, filename)
        self.perform_network_check()

    def read_network_operational_data_from_file(self, operational_data_filename):
        filename = os.path.join(self.data_dir, self.name, operational_data_filename)
        data = _read_network_operational_data_from_file(self, filename)
        _update_network_with_excel_data(self, data)

    def read_network_parameters(self, params_filename):
        filename = os.path.join(self.data_dir, self.name, params_filename)
        self.params.read_parameters_from_file(filename)

    def node_exists(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return True
        return False

    def perform_network_check(self):
        _perform_network_check(self)


# ======================================================================================================================
#  NETWORK DATA read functions
# ======================================================================================================================
def _read_network_data(network, operational_data_filename):

    # Read info from file(s)
    network.read_network_from_json_file(f'{network.name}.json')
    network.read_network_operational_data_from_file(operational_data_filename)

    if network.params.print_to_screen:
        network.network.print_network_to_screen()
    if network.params.plot_diagram:
        network.network.plot_diagram()


def _read_network_from_json_file(network, filename):

    network_data = convert_json_to_dict(read_json_file(filename))

    # Network base
    network.baseMVA = float(network_data['baseMVA'])

    # Nodes
    for node_data in network_data['nodes']:
        node = Node()
        node.bus_i = int(node_data['bus_i'])
        node.type = int(node_data['type'])
        node.gs = float(node_data['Gs']) / network.baseMVA
        node.bs = float(node_data['Bs']) / network.baseMVA
        node.base_kv = float(node_data['baseKV'])
        node.v_max = float(node_data['Vmax'])
        node.v_min = float(node_data['Vmin'])
        network.nodes.append(node)

    # Generators
    for gen_data in network_data['generators']:
        generator = Generator()
        generator.gen_id = int(gen_data['gen_id'])
        generator.bus = int(gen_data['bus'])
        if not network.node_exists(generator.bus):
            print(f'[ERROR] Generator {generator.gen_id}. Node {generator.bus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        generator.pmax = float(gen_data['Pmax']) / network.baseMVA
        generator.pmin = float(gen_data['Pmin']) / network.baseMVA
        generator.qmax = float(gen_data['Qmax']) / network.baseMVA
        generator.qmin = float(gen_data['Qmin']) / network.baseMVA
        generator.vg = float(gen_data['Vg'])
        generator.status = bool(gen_data['status'])
        gen_type = gen_data['type']
        if gen_type == 'REF':
            generator.gen_type = GEN_REFERENCE
        elif gen_type == 'CONV':
            generator.gen_type = GEN_CONV
        elif gen_type == 'PV':
            generator.gen_type = GEN_RES_SOLAR
        elif gen_type == 'WIND':
            generator.gen_type = GEN_RES_WIND
        elif gen_type == 'RES_OTHER':
            generator.gen_type = GEN_RES_OTHER
        elif gen_type == 'RES_CONTROLLABLE':
            generator.gen_type = GEN_RES_CONTROLLABLE
        if 'pf_control' in gen_data:
            generator.power_factor_control = bool(gen_data['pf_control'])
            if 'pf_max' in gen_data:
                generator.max_pf = float(gen_data['pf_max'])
            if 'pf_min' in gen_data:
                generator.min_pf = float(gen_data['pf_min'])
        network.generators.append(generator)

    # Loads
    for load_data in network_data['loads']:
        load = Load()
        load.load_id = int(load_data['load_id'])
        load.bus = int(load_data['bus'])
        if not network.node_exists(load.bus):
            print(f'[ERROR] Load {load.load_id }. Node {load.bus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        load.status = bool(load_data['status'])
        load.fl_reg = bool(load_data['fl_reg'])
        network.loads.append(load)

    # Lines
    for line_data in network_data['lines']:
        branch = Branch()
        branch.branch_id = int(line_data['branch_id'])
        branch.fbus = int(line_data['fbus'])
        if not network.node_exists(branch.fbus):
            print(f'[ERROR] Line {branch.branch_id }. Node {branch.fbus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        branch.tbus = int(line_data['tbus'])
        if not network.node_exists(branch.tbus):
            print(f'[ERROR] Line {branch.branch_id }. Node {branch.tbus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        branch.r = float(line_data['r'])
        branch.x = float(line_data['x'])
        branch.b_sh = float(line_data['b'])
        branch.rate = float(line_data['rating'])
        branch.status = bool(line_data['status'])
        network.branches.append(branch)

    # Transformers
    if 'transformers' in network_data:
        for transf_data in network_data['transformers']:
            branch = Branch()
            branch.branch_id = int(transf_data['branch_id'])
            branch.fbus = int(transf_data['fbus'])
            if not network.node_exists(branch.fbus):
                print(f'[ERROR] Transformer {branch.branch_id}. Node {branch.fbus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            branch.tbus = int(transf_data['tbus'])
            if not network.node_exists(branch.tbus):
                print(f'[ERROR] Transformer {branch.branch_id}. Node {branch.tbus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            branch.r = float(transf_data['r'])
            branch.x = float(transf_data['x'])
            branch.b_sh = float(transf_data['b'])
            branch.rate = float(transf_data['rating'])
            branch.ratio = float(transf_data['ratio'])
            branch.status = bool(transf_data['status'])
            branch.is_transformer = True
            branch.vmag_reg = bool(transf_data['vmag_reg'])
            network.branches.append(branch)

    # Energy Storages
    if 'energy_storages' in network_data:
        for energy_storage_data in network_data['energy_storages']:
            energy_storage = EnergyStorage()
            energy_storage.es_id = int(energy_storage_data['es_id'])
            energy_storage.bus = int(energy_storage_data['bus'])
            if not network.node_exists(energy_storage.bus):
                print(f'[ERROR] Energy Storage {energy_storage.es_id}. Node {energy_storage.bus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            energy_storage.s = float(energy_storage_data['s']) / network.baseMVA
            energy_storage.e = float(energy_storage_data['e']) / network.baseMVA
            energy_storage.e_init = energy_storage.e * ENERGY_STORAGE_RELATIVE_INIT_SOC
            energy_storage.e_min = energy_storage.e * ENERGY_STORAGE_MIN_ENERGY_STORED
            energy_storage.e_max = energy_storage.e * ENERGY_STORAGE_MAX_ENERGY_STORED
            energy_storage.eff_ch = float(energy_storage_data['eff_ch'])
            energy_storage.eff_dch = float(energy_storage_data['eff_dch'])
            energy_storage.max_pf = float(energy_storage_data['max_pf'])
            energy_storage.min_pf = float(energy_storage_data['min_pf'])
            network.energy_storages.append(energy_storage)


# ======================================================================================================================
#   Other (aux) functions
# ======================================================================================================================
def _perform_network_check(network):

    n_bus = len(network.nodes)
    if n_bus == 0:
        print(f'[ERROR] Reading network {network.name}. No nodes imported.')
        exit(ERROR_NETWORK_FILE)

    n_branch = len(network.branches)
    if n_branch == 0:
        print(f'[ERROR] Reading network {network.name}. No branches imported.')
        exit(ERROR_NETWORK_FILE)
