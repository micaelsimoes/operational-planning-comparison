import os
from idlelib.pyparse import trans

import pandas as pd
from math import isclose
from copy import copy
import pyomo.opt as po
import pyomo.environ as pe
from pyomo.scripting.util import process_results

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

    def run_hierarchical_coordination(self, t=0, num_steps=8, print_pq_map=False):
        _run_hierarchical_coordination(self, t, num_steps, print_pq_map)

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
    #_add_shared_energy_storage_to_transmission_network(operational_planning)
    #_add_shared_energy_storage_to_distribution_network(operational_planning)


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
def _run_hierarchical_coordination(operational_planning, t, num_steps, print_pq_map):

    print('[INFO] Running HIERARCHICAL OPERATIONAL PLANNING...')

    transmission_network = operational_planning.transmission_network
    distribution_networks = operational_planning.distribution_networks
    results = {'tso': dict(), 'dso': dict()}

    # Get DN models representation (PQ maps)
    dn_models = dict()
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        init_solution, ineqs = distribution_network.determine_pq_map(t=t, num_steps=num_steps, print_pq_map=print_pq_map)
        dn_models[node_id] = {
            'initial_solution': init_solution,
            'inequalities': ineqs
        }

    # TN model
    tn_model = transmission_network.build_model(t=t)
    tn_model.active_distribution_networks = range(len(transmission_network.active_distribution_network_nodes))

    # TN, Fix Pc, Qc at the interface nodes, free flexibility
    for dn in tn_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_load_idx = transmission_network.get_adn_load_idx(adn_node_id)
        init_solution = dn_models[adn_node_id]['initial_solution']
        for s_o in tn_model.scenarios_operation:
            tn_model.pc[adn_load_idx, s_o].setub(init_solution['Pg'] / transmission_network.baseMVA + EQUALITY_TOLERANCE)
            tn_model.pc[adn_load_idx, s_o].setlb(init_solution['Pg'] / transmission_network.baseMVA - EQUALITY_TOLERANCE)
            tn_model.qc[adn_load_idx, s_o].setub(init_solution['Qg'] / transmission_network.baseMVA + EQUALITY_TOLERANCE)
            tn_model.qc[adn_load_idx, s_o].setlb(init_solution['Qg'] / transmission_network.baseMVA - EQUALITY_TOLERANCE)
            tn_model.flex_p_up[adn_load_idx, s_o].setub(None)
            tn_model.flex_p_down[adn_load_idx, s_o].setub(None)
            tn_model.flex_q_up[adn_load_idx, s_o].setub(None)
            tn_model.flex_q_down[adn_load_idx, s_o].setub(None)
            if transmission_network.params.l_curt:
                tn_model.pc_curt_down[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tn_model.pc_curt_up[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tn_model.qc_curt_down[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tn_model.qc_curt_up[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)

    # TN, Add expected interface values
    tn_model.expected_interface_pf_p = pe.Var(tn_model.active_distribution_networks, domain=pe.Reals, initialize=0.00)
    tn_model.expected_interface_pf_q = pe.Var(tn_model.active_distribution_networks, domain=pe.Reals, initialize=0.00)
    tn_model.interface_expected_values = pe.ConstraintList()
    for dn in tn_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_load_idx = transmission_network.get_adn_load_idx(adn_node_id)
        expected_pf_p = 0.00
        expected_pf_q = 0.00
        for s_o in tn_model.scenarios_operation:
            omega_oper = transmission_network.prob_operation_scenarios[s_o]
            adn_pc = tn_model.pc[adn_load_idx, s_o] + tn_model.flex_p_up[adn_load_idx, s_o] - tn_model.flex_p_down[adn_load_idx, s_o]
            adn_qc = tn_model.qc[adn_load_idx, s_o] + tn_model.flex_q_up[adn_load_idx, s_o] - tn_model.flex_q_down[adn_load_idx, s_o]
            expected_pf_p += omega_oper * adn_pc
            expected_pf_q += omega_oper * adn_qc
        tn_model.interface_expected_values.add(tn_model.expected_interface_pf_p[dn] <= expected_pf_p + SMALL_TOLERANCE)
        tn_model.interface_expected_values.add(tn_model.expected_interface_pf_p[dn] >= expected_pf_p - SMALL_TOLERANCE)
        tn_model.interface_expected_values.add(tn_model.expected_interface_pf_q[dn] <= expected_pf_q + SMALL_TOLERANCE)
        tn_model.interface_expected_values.add(tn_model.expected_interface_pf_q[dn] >= expected_pf_q - SMALL_TOLERANCE)

    # TN, Add ADNs' PQ maps constraints
    tn_model.pq_maps = pe.ConstraintList()
    for dn in tn_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_pq_map = dn_models[adn_node_id]
        initial_solution = adn_pq_map['initial_solution']
        for ineq in adn_pq_map['inequalities']:
            a = ineq['Pg']
            b = ineq['Qg']
            c = ineq['c'] / transmission_network.baseMVA
            tn_model.pq_maps.add(a * tn_model.expected_interface_pf_p[dn] + b * tn_model.expected_interface_pf_q[dn] <= c)
            # tn_model.pq_maps.add(tn_model.expected_interface_pf_p[dn] <= initial_solution['Pg'] / transmission_network.baseMVA * 1.10)
            # tn_model.pq_maps.add(tn_model.expected_interface_pf_p[dn] >= initial_solution['Pg'] / transmission_network.baseMVA * 0.90)
            # tn_model.pq_maps.add(tn_model.expected_interface_pf_q[dn] <= initial_solution['Qg'] / transmission_network.baseMVA * 1.10)
            # tn_model.pq_maps.add(tn_model.expected_interface_pf_q[dn] >= initial_solution['Qg'] / transmission_network.baseMVA * 0.90)

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    obj = copy(tn_model.objective.expr)
    tn_model.penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
    tn_model.penalty_regularization.fix(PENALTY_REGULARIZATION)
    for dn in tn_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_load_idx = transmission_network.get_adn_load_idx(adn_node_id)
        for s_o in tn_model.scenarios_operation:
            obj += tn_model.penalty_regularization * transmission_network.baseMVA * (tn_model.pc[adn_load_idx, s_o] - tn_model.expected_interface_pf_p[dn]) ** 2
            obj += tn_model.penalty_regularization * transmission_network.baseMVA * (tn_model.qc[adn_load_idx, s_o] - tn_model.expected_interface_pf_q[dn]) ** 2
    tn_model.objective.expr = obj

    results = transmission_network.optimize(tn_model)
    process_results = transmission_network.process_results(tn_model, results)
    transmission_network.write_optimization_results_to_excel(process_results, filename=f'{transmission_network.name}_debug')


def create_distribution_networks_models(distribution_networks):
    dso_models = dict()
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        dso_model = distribution_network.build_model()
        dso_models[node_id] = dso_model
    return dso_models

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
