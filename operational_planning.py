import os
import pandas as pd
from math import isclose
from network import Network
from energy_storage import EnergyStorage
from shared_energy_storage_data import SharedEnergyStorageData
from operational_planning_parameters import ADMMParameters
from helper_functions import *


# ======================================================================================================================
#   Class OPERATIONAL PLANNING
# ======================================================================================================================
class OperationalPlanning:

    def __init__(self, data_dir, filename):
        self.name = filename.replace('.json', '')
        self.data_dir = data_dir
        self.filename = filename
        self.results_dir = os.path.join(data_dir, 'Results')
        self.diagrams_dir = os.path.join(data_dir, 'Diagrams')
        self.cost_energy_p = 0.00
        self.cost_flex = 0.00
        self.distribution_networks = dict()
        self.transmission_network = Network()
        self.shared_ess_data = SharedEnergyStorageData()
        self.active_distribution_network_nodes = list()
        self.params = ADMMParameters()

    def run_hierarchical_coordination(self):
        _run_hierarchical_coordination(self)

    def read_case_study(self):
        _read_case_study(self)

    def read_market_data_from_file(self, filename):
        _read_market_data_from_file(self, filename)

    def read_operational_planning_parameters_from_file(self, params_file):
        filename = os.path.join(self.data_dir, params_file)
        self.params.read_parameters_from_file(filename)


# ======================================================================================================================
#  OPERATIONAL PLANNING PROBLEM read functions
# ======================================================================================================================
def _read_case_study(operational_planning):

    # Create results folder
    if not os.path.exists(operational_planning.results_dir):
        os.makedirs(operational_planning.results_dir)

    # Create diagrams folder
    if not os.path.exists(operational_planning.diagrams_dir):
        os.makedirs(operational_planning.diagrams_dir)

    # Read specification file
    filename = os.path.join(operational_planning.data_dir, operational_planning.filename)
    operational_planning_data = convert_json_to_dict(read_json_file(filename))

    # Market data
    market_data_filename = operational_planning_data['MarketData']['market_data_filename']
    operational_planning.read_market_data_from_file(market_data_filename)

    # Distribution Networks
    for distribution_network_data in operational_planning_data['DistributionNetworks']:

        print('[INFO] Reading DISTRIBUTION NETWORK DATA from file(s)...')

        network_name = distribution_network_data['name']                                    # Network filename
        params_filename = distribution_network_data['params_filename']                      # Params filename
        operational_data_filename = distribution_network_data['operational_data_filename']  # Operational data filename
        connection_nodeid = distribution_network_data['connection_node_id']                 # Connection node ID

        distribution_network = Network()
        distribution_network.name = network_name
        distribution_network.is_transmission = False
        distribution_network.data_dir = operational_planning.data_dir
        distribution_network.results_dir = operational_planning.results_dir
        distribution_network.diagrams_dir = operational_planning.diagrams_dir
        distribution_network.cost_energy_p = operational_planning.cost_energy_p
        distribution_network.cost_flex = operational_planning.cost_flex
        distribution_network.read_network_parameters(params_filename)
        distribution_network.read_network_data(operational_data_filename)
        distribution_network.tn_connection_nodeid = connection_nodeid
        operational_planning.distribution_networks[connection_nodeid] = distribution_network
    operational_planning.active_distribution_network_nodes = [node_id for node_id in operational_planning.distribution_networks]

    # Transmission Network
    print('[INFO] Reading TRANSMISSION NETWORK DATA from file(s)...')
    transmission_network = Network()
    transmission_network.name = operational_planning_data['TransmissionNetwork']['name']
    transmission_network.is_transmission = True
    transmission_network.data_dir = operational_planning.data_dir
    transmission_network.results_dir = operational_planning.results_dir
    transmission_network.diagrams_dir = operational_planning.diagrams_dir
    transmission_network.cost_energy_p = operational_planning.cost_energy_p
    transmission_network.cost_flex = operational_planning.cost_flex
    params_filename = operational_planning_data['TransmissionNetwork']['params_filename']
    transmission_network.read_network_parameters(params_filename)
    operational_data_filename = operational_planning_data['TransmissionNetwork']['operational_data_filename']
    transmission_network.read_network_data(operational_data_filename)
    transmission_network.active_distribution_network_nodes = [node_id for node_id in operational_planning.distribution_networks]
    operational_planning.transmission_network = transmission_network

    # Shared ESS
    print('[INFO] Reading SHARED ESS DATA from file(s)...')
    shared_ess_data = SharedEnergyStorageData()
    shared_ess_data.data_dir = operational_planning.data_dir
    shared_ess_data_filename = operational_planning_data['SharedEnergyStorages']['shared_ess_filename']
    shared_ess_data.read_shared_energy_storage_data_from_file(shared_ess_data_filename)
    operational_planning.shared_ess_data = shared_ess_data

    # Planning Parameters
    print(f'[INFO] Reading PLANNING PARAMETERS from file...')
    params_filename = operational_planning_data['PlanningParameters']['params_filename']
    operational_planning.read_operational_planning_parameters_from_file(params_filename)

    _check_interface_nodes_base_voltage_consistency(operational_planning)

    # Add Shared Energy Storages to Transmission and Distribution Networks
    _add_shared_energy_storage_to_transmission_network(operational_planning)
    _add_shared_energy_storage_to_distribution_network(operational_planning)



# ======================================================================================================================
#  MARKET DATA read functions
# ======================================================================================================================
def _read_market_data_from_file(operational_planning, market_data_filename):
    try:
        filename = os.path.join(operational_planning.data_dir, 'Market Data', market_data_filename)
        market_costs = _get_market_costs_from_excel_file(filename)
        operational_planning.cost_energy_p = market_costs['Cp']
        operational_planning.cost_flex = market_costs['Cflex']
    except:
        print(f'[ERROR] Reading market data from file(s). Exiting...')
        exit(ERROR_SPECIFICATION_FILE)


def _get_market_costs_from_excel_file(filename):
    data = pd.read_excel(filename)
    num_rows, num_cols = data.shape
    market_costs = dict()
    for i in range(num_rows):
        cost_type = data.iloc[i, 0]
        values = list()
        for j in range(num_cols - 1):
            values.append(data.iloc[i, j + 1])
        market_costs[cost_type] = values
    return market_costs


# ======================================================================================================================
#  OPERATIONAL PLANNING functions
# ======================================================================================================================
def _run_hierarchical_coordination(operational_planning):

    print('[INFO] Running HIERARCHICAL OPERATIONAL PLANNING...')

    print()


# ======================================================================================================================
#   Aux functions
# ======================================================================================================================
def _check_interface_nodes_base_voltage_consistency(operational_planning):
    for node_id in operational_planning.distribution_networks:
        tn_node_base_kv = operational_planning.transmission_network.get_node_base_kv(node_id)
        dn_ref_node_id = operational_planning.distribution_networks[node_id].get_reference_node_id()
        dn_node_base_kv = operational_planning.distribution_networks[node_id].get_node_base_kv(dn_ref_node_id)
        if not isclose(tn_node_base_kv, dn_node_base_kv, rel_tol=5e-2):
            print(f'[ERROR] Inconsistent TN-DN base voltage at node {node_id}! Check network(s). Exiting')
            exit(ERROR_SPECIFICATION_FILE)


def _add_shared_energy_storage_to_transmission_network(operational_planning):
    s_base = operational_planning.transmission_network.baseMVA
    for node_id in operational_planning.active_distribution_network_nodes:
        for shared_ess in operational_planning.shared_ess_data.shared_energy_storages:
            if shared_ess.bus == node_id:
                shared_energy_storage = EnergyStorage()
                shared_energy_storage.bus = node_id
                shared_energy_storage.s = shared_ess.s / s_base
                shared_energy_storage.e = shared_ess.e / s_base
                shared_energy_storage.e_init = shared_ess.e_init / s_base
                shared_energy_storage.e_min = shared_ess.e_min / s_base
                shared_energy_storage.e_max = shared_ess.e_max / s_base
                shared_energy_storage.eff_ch = shared_ess.eff_ch
                shared_energy_storage.eff_dch = shared_ess.eff_dch
                shared_energy_storage.max_pf = shared_ess.max_pf
                shared_energy_storage.min_pf = shared_ess.min_pf
                operational_planning.transmission_network.shared_energy_storages.append(shared_energy_storage)


def _add_shared_energy_storage_to_distribution_network(operational_planning):
    for node_id in operational_planning.distribution_networks:
        s_base = operational_planning.distribution_networks[node_id].baseMVA
        for shared_ess in operational_planning.shared_ess_data.shared_energy_storages:
            if shared_ess.bus == node_id:
                shared_energy_storage = EnergyStorage()
                shared_energy_storage.bus = operational_planning.distribution_networks[node_id].get_reference_node_id()
                shared_energy_storage.s = shared_energy_storage.s / s_base
                shared_energy_storage.e = shared_energy_storage.e / s_base
                shared_energy_storage.e_init = shared_ess.e_init / s_base
                shared_energy_storage.e_min = shared_ess.e_min / s_base
                shared_energy_storage.e_max = shared_ess.e_max / s_base
                shared_energy_storage.eff_ch = shared_ess.eff_ch
                shared_energy_storage.eff_dch = shared_ess.eff_dch
                shared_energy_storage.max_pf = shared_ess.max_pf
                shared_energy_storage.min_pf = shared_ess.min_pf
                operational_planning.distribution_networks[node_id].shared_energy_storages.append(shared_energy_storage)
