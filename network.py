import os
import pandas as pd
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

    def get_num_renewable_gens(self):
        num_renewable_gens = 0
        for generator in self.generators:
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                num_renewable_gens += 1
        return num_renewable_gens

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
#   NETWORK OPERATIONAL DATA read functions
# ======================================================================================================================
def _read_network_operational_data_from_file(network, filename):

    data = {
        'consumption': {
            'pc': dict(), 'qc': dict()
        },
        'flexibility': {
            'upward': dict(),
            'downward': dict()
        },
        'generation': {
            'pg': dict(), 'qg': dict(), 'status': list()
        }
    }

    # Scenario information
    num_oper_scenarios, prob_oper_scenarios = _get_operational_scenario_info_from_excel_file(filename, 'Main')
    network.prob_operation_scenarios = prob_oper_scenarios

    # Consumption and Generation data -- by scenario
    for i in range(len(network.prob_operation_scenarios)):

        sheet_name_pc = f'Pc, S{i + 1}'
        sheet_name_qc = f'Qc, S{i + 1}'
        sheet_name_pg = f'Pg, S{i + 1}'
        sheet_name_qg = f'Qg, S{i + 1}'

        # Consumption per scenario (active, reactive power)
        pc_scenario = _get_consumption_flexibility_data_from_excel_file(filename, sheet_name_pc)
        qc_scenario = _get_consumption_flexibility_data_from_excel_file(filename, sheet_name_qc)
        if not pc_scenario:
            print(f'[ERROR] Network {network.name}. No active power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        if not qc_scenario:
            print(f'[ERROR] Network {network.name}. No reactive power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        data['consumption']['pc'][i] = pc_scenario
        data['consumption']['qc'][i] = qc_scenario

        # Generation per scenario (active, reactive power)
        num_renewable_gens = network.get_num_renewable_gens()
        if num_renewable_gens > 0:
            pg_scenario = _get_generation_data_from_excel_file(filename, sheet_name_pg)
            qg_scenario = _get_generation_data_from_excel_file(filename, sheet_name_qg)
            if not pg_scenario:
                print(f'[ERROR] Network {network.name}. No active power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            if not qg_scenario:
                print(f'[ERROR] Network {network.name}. No reactive power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            data['generation']['pg'][i] = pg_scenario
            data['generation']['qg'][i] = qg_scenario

    # Generators status. Note: common to all scenarios
    data['generation']['status'] = _get_generator_status_from_excel_file(filename, f'GenStatus')

    # Flexibility data
    flex_up_p = _get_consumption_flexibility_data_from_excel_file(filename, f'UpFlex')
    if not flex_up_p:
        for load in network.loads:
            flex_up_p[load.load_id] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['upward'] = flex_up_p

    flex_down_p = _get_consumption_flexibility_data_from_excel_file(filename, f'DownFlex')
    if not flex_down_p:
        for load in network.loads:
            flex_down_p[load.load_id] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['downward'] = flex_down_p

    return data


def _get_operational_scenario_info_from_excel_file(filename, sheet_name):

    num_scenarios = 0
    prob_scenarios = list()

    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        if is_int(df.iloc[0, 1]):
            num_scenarios = int(df.iloc[0, 1])
        else:
            print(f'[ERROR] File {filename}. Num scenarios should be an int!')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        for i in range(num_scenarios):
            if is_float(df.iloc[0, i+2]):
                prob_scenarios.append(float(df.iloc[0, i+2]))
            else:
                print(f'[ERROR] File {filename}. Scenario probability should be a float!')
                exit(ERROR_OPERATIONAL_DATA_FILE)
    except:
        print(f'[ERROR] Workbook {filename}. Sheet {sheet_name} does not exist.')
        exit(ERROR_OPERATIONAL_DATA_FILE)

    if num_scenarios != len(prob_scenarios):
        print(f'[WARNING] Workbook {filename}. Data file. Number of scenarios different from the probability vector!')

    if round(sum(prob_scenarios), 2) != 1.00:
        print(f'[ERROR] Workbook {filename}. Probability of scenarios does not add up to 100%.')
        exit(ERROR_OPERATIONAL_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_consumption_flexibility_data_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        processed_data = dict()
        for i in range(num_rows):
            node_id = data.iloc[i, 0]
            processed_data[node_id] = [0.0 for _ in range(num_cols - 1)]
        for node_id in processed_data:
            node_values = [0.0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == node_id:
                    for j in range(0, num_cols - 1):
                        node_values[j] += data.iloc[i, j + 1]
            processed_data[node_id] = node_values
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        processed_data = {}

    return processed_data


def _get_generation_data_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        processed_data = dict()
        for i in range(num_rows):
            gen_id = data.iloc[i, 0]
            processed_data[gen_id] = [0.0 for _ in range(num_cols - 1)]
        for gen_id in processed_data:
            processed_data_gen = [0.0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == gen_id:
                    for j in range(0, num_cols - 1):
                        processed_data_gen[j] += data.iloc[i, j + 1]
            processed_data[gen_id] = processed_data_gen
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        processed_data = {}

    return processed_data


def _get_generator_status_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        status_values = dict()
        for i in range(num_rows):
            gen_id = data.iloc[i, 0]
            status_values[gen_id] = list()
            for j in range(0, num_cols - 1):
                status_values[gen_id].append(bool(data.iloc[i, j + 1]))
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        status_values = list()

    return status_values


def _update_network_with_excel_data(network, data):

    for load in network.loads:

        load_id = load.load_id
        load.pd = dict()         # Note: Changes Pd and Qd fields to dicts (per scenario)
        load.qd = dict()

        for s in range(len(network.prob_operation_scenarios)):
            pc = _get_consumption_from_data(data, load_id, network.num_instants, s, DATA_ACTIVE_POWER)
            qc = _get_consumption_from_data(data, load_id, network.num_instants, s, DATA_REACTIVE_POWER)
            load.pd[s] = [instant / network.baseMVA for instant in pc]
            load.qd[s] = [instant / network.baseMVA for instant in qc]
        flex_up_p = _get_flexibility_from_data(data, load_id, network.num_instants, DATA_UPWARD_FLEXIBILITY)
        flex_down_p = _get_flexibility_from_data(data, load_id, network.num_instants, DATA_DOWNWARD_FLEXIBILITY)
        load.flexibility.upward = [p / network.baseMVA for p in flex_up_p]
        load.flexibility.downward = [q / network.baseMVA for q in flex_down_p]

    for generator in network.generators:

        generator.pg = dict()  # Note: Changes Pg and Qg fields to dicts (per scenario)
        generator.qg = dict()

        # Active and Reactive power
        for s in range(len(network.prob_operation_scenarios)):
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                pg = _get_generation_from_data(data, generator.gen_id, s, DATA_ACTIVE_POWER)
                qg = _get_generation_from_data(data, generator.gen_id, s, DATA_REACTIVE_POWER)
                generator.pg[s] = [instant / network.baseMVA for instant in pg]
                generator.qg[s] = [instant / network.baseMVA for instant in qg]
            else:
                generator.pg[s] = [0.00 for _ in range(network.num_instants)]
                generator.qg[s] = [0.00 for _ in range(network.num_instants)]

        # Status
        if generator.gen_id in data['generation']['status']:
            generator.status = data['generation']['status'][generator.gen_id]
        else:
            generator.status = [generator.status for _ in range(network.num_instants)]

    network.data_loaded = True


def _get_consumption_from_data(data, node_id, num_instants, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pc'
    else:
        power_label = 'qc'

    for node in data['consumption'][power_label][idx_scenario]:
        if node == node_id:
            return data['consumption'][power_label][idx_scenario][node_id]

    consumption = [0.0 for _ in range(num_instants)]

    return consumption


def _get_flexibility_from_data(data, node_id, num_instants, flex_type):

    flex_label = str()

    if flex_type == DATA_UPWARD_FLEXIBILITY:
        flex_label = 'upward'
    elif flex_type == DATA_DOWNWARD_FLEXIBILITY:
        flex_label = 'downward'
    elif flex_type == DATA_COST_FLEXIBILITY:
        flex_label = 'cost'
    else:
        print('[ERROR] Unrecognized flexibility type in get_flexibility_from_data. Exiting.')
        exit(1)

    for node in data['flexibility'][flex_label]:
        if node == node_id:
            return data['flexibility'][flex_label][node_id]

    flex = [0.0 for _ in range(num_instants)]   # Returns empty flexibility vector

    return flex


def _get_generation_from_data(data, gen_id, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pg'
    else:
        power_label = 'qg'

    return data['generation'][power_label][idx_scenario][gen_id]


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
