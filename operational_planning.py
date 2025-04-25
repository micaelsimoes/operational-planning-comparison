import os
import time
import pandas as pd
from math import isclose
from copy import copy
import pyomo.opt as po
import pyomo.environ as pe
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

    def run_distributed_coordination(self, t=0, consider_shared_ess=False):
        _run_distributed_coordination(self, t, consider_shared_ess)

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
#  HIERARCHICAL OPERATIONAL PLANNING functions
# ======================================================================================================================
def _run_hierarchical_coordination(operational_planning, t, num_steps, print_pq_map):

    print('[INFO] Running HIERARCHICAL OPERATIONAL PLANNING...')

    transmission_network = operational_planning.transmission_network
    distribution_networks = operational_planning.distribution_networks

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
        # initial_solution = adn_pq_map['initial_solution']
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


# ======================================================================================================================
#  DISTRIBUTED OPERATIONAL PLANNING functions
# ======================================================================================================================
def _run_distributed_coordination(operational_planning, t, consider_shared_ess):

    print('[INFO] Running DISTRIBUTED OPERATIONAL PLANNING...')

    transmission_network = operational_planning.transmission_network
    distribution_networks = operational_planning.distribution_networks
    admm_parameters = operational_planning.params
    results = {'tso': dict(), 'dso': dict()}

    # ------------------------------------------------------------------------------------------------------------------
    # 0. Initialization

    print('[INFO]\t\t - Initializing...')

    start = time.time()
    from_warm_start = False
    primal_evolution = list()

    # Create ADMM variables
    consensus_vars, dual_vars = create_admm_variables(operational_planning)

    # Create ADN models, get initial power flows
    dso_models, results['dso'] = create_distribution_networks_models(distribution_networks, consensus_vars, t, consider_shared_ess)
    tso_model, results['tso'] = create_transmission_network_model(transmission_network, consensus_vars, t, consider_shared_ess)


def create_transmission_network_model(transmission_network, consensus_vars, t, consider_shared_ess):

    # Build model
    tso_model = transmission_network.build_model(t)
    s_base = transmission_network.baseMVA

    # Update model with expected interface values
    tso_model.active_distribution_networks = range(len(transmission_network.active_distribution_network_nodes))

    # Free Vmag, Pc, Qc at the interface nodes
    for dn in tso_model[year][day].active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_node_idx = transmission_network.network[year][day].get_node_idx(adn_node_id)
        adn_load_idx = transmission_network.network[year][day].get_adn_load_idx(adn_node_id)
        _, v_max = transmission_network.network[year][day].get_node_voltage_limits(adn_node_id)
        for s_m in tso_model[year][day].scenarios_market:
            for s_o in tso_model[year][day].scenarios_operation:
                for p in tso_model[year][day].periods:

                    tso_model[year][day].e[adn_node_idx, s_m, s_o, p].fixed = False
                    tso_model[year][day].e[adn_node_idx, s_m, s_o, p].setub(v_max + SMALL_TOLERANCE)
                    tso_model[year][day].e[adn_node_idx, s_m, s_o, p].setlb(-v_max - SMALL_TOLERANCE)
                    tso_model[year][day].f[adn_node_idx, s_m, s_o, p].fixed = False
                    tso_model[year][day].f[adn_node_idx, s_m, s_o, p].setub(v_max + SMALL_TOLERANCE)
                    tso_model[year][day].f[adn_node_idx, s_m, s_o, p].setlb(-v_max - SMALL_TOLERANCE)
                    if transmission_network.params.slacks.grid_operation.voltage:
                        tso_model[year][day].slack_e[adn_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                        tso_model[year][day].slack_e[adn_node_idx, s_m, s_o, p].setlb(-EQUALITY_TOLERANCE)
                        tso_model[year][day].slack_f[adn_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                        tso_model[year][day].slack_f[adn_node_idx, s_m, s_o, p].setlb(-EQUALITY_TOLERANCE)

                    tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].fixed = False
                    tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setub(None)
                    tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setlb(None)
                    tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].fixed = False
                    tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setub(None)
                    tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setlb(None)
                    if transmission_network.params.fl_reg:
                        tso_model[year][day].flex_p_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                        tso_model[year][day].flex_p_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                    if transmission_network.params.l_curt:
                        tso_model[year][day].pc_curt_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                        tso_model[year][day].pc_curt_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                        tso_model[year][day].qc_curt_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                        tso_model[year][day].qc_curt_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)

    # Add expected interface and shared ESS values
    tso_model[year][day].expected_interface_vmag_sqr = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.NonNegativeReals, initialize=1.00)
    tso_model[year][day].expected_interface_pf_p = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
    tso_model[year][day].expected_interface_pf_q = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
    tso_model[year][day].expected_shared_ess_p = pe.Var(tso_model[year][day].shared_energy_storages, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
    tso_model[year][day].expected_shared_ess_q = pe.Var(tso_model[year][day].shared_energy_storages, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
    tso_model[year][day].interface_expected_values = pe.ConstraintList()
    for dn in tso_model[year][day].active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_node_idx = transmission_network.network[year][day].get_node_idx(adn_node_id)
        adn_load_idx = transmission_network.network[year][day].get_adn_load_idx(adn_node_id)
        for p in tso_model[year][day].periods:
            expected_vmag_sqr = 0.00
            expected_pf_p = 0.00
            expected_pf_q = 0.00
            for s_m in tso_model[year][day].scenarios_market:
                omega_market = transmission_network.network[year][day].prob_market_scenarios[s_m]
                for s_o in tso_model[year][day].scenarios_operation:
                    omega_oper = transmission_network.network[year][day].prob_operation_scenarios[s_o]
                    expected_vmag_sqr += omega_market * omega_oper * (tso_model[year][day].e[adn_node_idx, s_m, s_o, p] ** 2 + tso_model[year][day].f[adn_node_idx, s_m, s_o, p] ** 2)
                    expected_pf_p += omega_market * omega_oper * tso_model[year][day].pc[adn_load_idx, s_m, s_o, p]
                    expected_pf_q += omega_market * omega_oper * tso_model[year][day].qc[adn_load_idx, s_m, s_o, p]
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_interface_vmag_sqr[dn, p] <= expected_vmag_sqr + SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_interface_vmag_sqr[dn, p] >= expected_vmag_sqr - SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_interface_pf_p[dn, p] <= expected_pf_p + SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_interface_pf_p[dn, p] >= expected_pf_p - SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_interface_pf_q[dn, p] <= expected_pf_q + SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_interface_pf_q[dn, p] >= expected_pf_q - SMALL_TOLERANCE)
    for e in tso_model[year][day].shared_energy_storages:
        for p in tso_model[year][day].periods:
            expected_ess_p = 0.00
            expected_ess_q = 0.00
            for s_m in tso_model[year][day].scenarios_market:
                omega_market = transmission_network.network[year][day].prob_market_scenarios[s_m]
                for s_o in tso_model[year][day].scenarios_operation:
                    omega_oper = transmission_network.network[year][day].prob_operation_scenarios[s_o]
                    expected_ess_p += omega_market * omega_oper * tso_model[year][day].shared_es_pnet[e, s_m, s_o, p]
                    expected_ess_q += omega_market * omega_oper * tso_model[year][day].shared_es_qnet[e, s_m, s_o, p]
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_shared_ess_p[e, p] <= expected_ess_p + SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_shared_ess_p[e, p] >= expected_ess_p - SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_shared_ess_q[e, p] <= expected_ess_q + SMALL_TOLERANCE)
            tso_model[year][day].interface_expected_values.add(tso_model[year][day].expected_shared_ess_q[e, p] >= expected_ess_q - SMALL_TOLERANCE)

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    obj = copy(tso_model[year][day].objective.expr)
    tso_model[year][day].penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
    tso_model[year][day].penalty_regularization.fix(PENALTY_REGULARIZATION)
    for dn in tso_model[year][day].active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_node_idx = transmission_network.network[year][day].get_node_idx(adn_node_id)
        adn_load_idx = transmission_network.network[year][day].get_adn_load_idx(adn_node_id)
        for s_m in tso_model[year][day].scenarios_market:
            for s_o in tso_model[year][day].scenarios_operation:
                for p in tso_model[year][day].periods:
                    obj += tso_model[year][day].penalty_regularization * ((tso_model[year][day].e[adn_node_idx, s_m, s_o, p] ** 2 + tso_model[year][day].f[adn_node_idx, s_m, s_o, p] ** 2) - tso_model[year][day].expected_interface_vmag_sqr[dn, p]) ** 2
                    obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].pc[adn_load_idx, s_m, s_o, p] - tso_model[year][day].expected_interface_pf_p[dn, p]) ** 2
                    obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].qc[adn_load_idx, s_m, s_o, p] - tso_model[year][day].expected_interface_pf_q[dn, p]) ** 2
    for e in tso_model[year][day].shared_energy_storages:
        for s_m in tso_model[year][day].scenarios_market:
            for s_o in tso_model[year][day].scenarios_operation:
                for p in tso_model[year][day].periods:
                    obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].shared_es_pnet[e, s_m, s_o, p] - tso_model[year][day].expected_shared_ess_p[e, p]) ** 2
                    obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].shared_es_qnet[e, s_m, s_o, p] - tso_model[year][day].expected_shared_ess_q[e, p]) ** 2
    tso_model[year][day].objective.expr = obj

    # Fix initial values, run OPF
    for year in transmission_network.years:
        for day in transmission_network.days:

            s_base = transmission_network.network[year][day].baseMVA

            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                v_base = transmission_network.network[year][day].get_node_base_kv(adn_node_id)
                for p in tso_model[year][day].periods:
                    interface_v_sqr = consensus_vars['v_sqr']['dso']['current'][adn_node_id][year][day][p] / (v_base ** 2)
                    interface_pf_p = consensus_vars['pf']['dso']['current'][adn_node_id][year][day]['p'][p] / s_base
                    interface_pf_q = consensus_vars['pf']['dso']['current'][adn_node_id][year][day]['q'][p] / s_base
                    tso_model[year][day].expected_interface_vmag_sqr[dn, p].fix(interface_v_sqr)
                    tso_model[year][day].expected_interface_pf_p[dn, p].fix(interface_pf_p)
                    tso_model[year][day].expected_interface_pf_q[dn, p].fix(interface_pf_q)

    # Run SMOPF
    results = transmission_network.optimize(tso_model)

    # Get initial interface and shared ESS values
    for year in transmission_network.years:
        for day in transmission_network.days:
            s_base = transmission_network.network[year][day].baseMVA
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                v_base = transmission_network.network[year][day].get_node_base_kv(adn_node_id)
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(adn_node_id)
                for p in tso_model[year][day].periods:
                    interface_v_sqr = pe.value(tso_model[year][day].expected_interface_vmag_sqr[dn, p]) * (v_base ** 2)
                    interface_pf_p = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                    interface_pf_q = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base
                    p_ess = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base
                    q_ess = pe.value(tso_model[year][day].expected_shared_ess_q[shared_ess_idx, p]) * s_base
                    consensus_vars['v_sqr']['tso']['current'][adn_node_id][year][day][p] = interface_v_sqr
                    consensus_vars['pf']['tso']['current'][adn_node_id][year][day]['p'][p] = interface_pf_p
                    consensus_vars['pf']['tso']['current'][adn_node_id][year][day]['q'][p] = interface_pf_q
                    consensus_vars['ess']['tso']['current'][adn_node_id][year][day]['p'][p] = p_ess
                    consensus_vars['ess']['tso']['current'][adn_node_id][year][day]['q'][p] = q_ess

    return tso_model, results



def create_distribution_networks_models(distribution_networks, consensus_vars, t, consider_shared_ess):

    dso_models = dict()
    results = dict()

    for node_id in distribution_networks:

        distribution_network = distribution_networks[node_id]
        s_base = distribution_network.baseMVA
        ref_node_id = distribution_network.get_reference_node_id()
        ref_node_idx = distribution_network.get_node_idx(ref_node_id)
        ref_gen_idx = distribution_network.get_reference_gen_idx()

        # Build model
        dso_model = distribution_network.build_model(t)

        # Update model with expected interface values
        dso_model.interface_expected_values = pe.ConstraintList()

        dso_model.expected_interface_vmag_sqr = pe.Var(domain=pe.NonNegativeReals, initialize=1.00)
        dso_model.expected_interface_pf_p = pe.Var(domain=pe.Reals, initialize=0.00)
        dso_model.expected_interface_pf_q = pe.Var(domain=pe.Reals, initialize=0.00)
        expected_vmag_sqr = 0.00
        expected_pf_p = 0.00
        expected_pf_q = 0.00
        for s_o in dso_model.scenarios_operation:
            omega_oper = distribution_network.prob_operation_scenarios[s_o]
            expected_vmag_sqr += omega_oper * dso_model.e[ref_node_idx, s_o] ** 2
            expected_pf_p += omega_oper * dso_model.pg[ref_gen_idx, s_o]
            expected_pf_q += omega_oper * dso_model.qg[ref_gen_idx, s_o]
        dso_model.interface_expected_values.add(dso_model.expected_interface_vmag_sqr <= expected_vmag_sqr + SMALL_TOLERANCE)
        dso_model.interface_expected_values.add(dso_model.expected_interface_vmag_sqr >= expected_vmag_sqr - SMALL_TOLERANCE)
        dso_model.interface_expected_values.add(dso_model.expected_interface_pf_p <= expected_pf_p + SMALL_TOLERANCE)
        dso_model.interface_expected_values.add(dso_model.expected_interface_pf_p >= expected_pf_p - SMALL_TOLERANCE)
        dso_model.interface_expected_values.add(dso_model.expected_interface_pf_q <= expected_pf_q + SMALL_TOLERANCE)
        dso_model.interface_expected_values.add(dso_model.expected_interface_pf_q >= expected_pf_q - SMALL_TOLERANCE)

        # Regularization -- Added to OF to minimize deviations from scenarios to expected values
        obj = copy(dso_model.objective.expr)
        dso_model.penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
        dso_model.penalty_regularization.fix(PENALTY_REGULARIZATION)
        for s_o in dso_model.scenarios_operation:
            obj += dso_model.penalty_regularization * (dso_model.e[ref_node_idx, s_o] ** 2 - dso_model.expected_interface_vmag_sqr) ** 2
            obj += dso_model.penalty_regularization * s_base * (dso_model.pg[ref_gen_idx, s_o] - dso_model.expected_interface_pf_p) ** 2
            obj += dso_model.penalty_regularization * s_base * (dso_model.qg[ref_gen_idx, s_o] - dso_model.expected_interface_pf_q) ** 2
        dso_model.objective.expr = obj

        if consider_shared_ess:
            shared_ess_idx = distribution_network.get_shared_energy_storage_idx(ref_node_id)
            dso_model.expected_shared_ess_p = pe.Var(domain=pe.Reals, initialize=0.00)
            dso_model.expected_shared_ess_q = pe.Var(domain=pe.Reals, initialize=0.00)
            expected_ess_p = 0.00
            expected_ess_q = 0.00
            for s_o in dso_model.scenarios_operation:
                omega_oper = distribution_network.prob_operation_scenarios[s_o]
                expected_ess_p += omega_oper * dso_model.shared_es_pnet[shared_ess_idx, s_o]
                expected_ess_q += omega_oper * dso_model.shared_es_qnet[shared_ess_idx, s_o]
            dso_model.interface_expected_values.add(dso_model.expected_shared_ess_p <= expected_ess_p + SMALL_TOLERANCE)
            dso_model.interface_expected_values.add(dso_model.expected_shared_ess_p >= expected_ess_p - SMALL_TOLERANCE)
            dso_model.interface_expected_values.add(dso_model.expected_shared_ess_q <= expected_ess_q + SMALL_TOLERANCE)
            dso_model.interface_expected_values.add(dso_model.expected_shared_ess_q >= expected_ess_q - SMALL_TOLERANCE)

            for s_o in dso_model.scenarios_operation:
                obj += dso_model.penalty_regularization * s_base * (dso_model.shared_es_pnet[shared_ess_idx, s_o] - dso_model.expected_shared_ess_p) ** 2
                obj += dso_model.penalty_regularization * s_base * (dso_model.shared_es_qnet[shared_ess_idx, s_o] - dso_model.expected_shared_ess_q) ** 2

        dso_model.objective.expr = obj

        # Run SMOPF
        results[node_id] = distribution_network.optimize(dso_model)

        # Get initial interface and shared ESS values
        v_base = distribution_network.get_node_base_kv(ref_node_id)
        interface_v_sqr = pe.value(dso_model.expected_interface_vmag_sqr) * (v_base ** 2)
        interface_pf_p = pe.value(dso_model.expected_interface_pf_p) * s_base
        interface_pf_q = pe.value(dso_model.expected_interface_pf_q) * s_base
        consensus_vars['v_sqr']['dso']['current'][node_id] = interface_v_sqr
        consensus_vars['pf']['dso']['current'][node_id]['p'] = interface_pf_p
        consensus_vars['pf']['dso']['current'][node_id]['q'] = interface_pf_q
        if consider_shared_ess:
            p_ess = pe.value(dso_model.expected_shared_ess_p) * s_base
            q_ess = pe.value(dso_model.expected_shared_ess_q) * s_base
            consensus_vars['ess']['dso']['current'][node_id]['p'] = p_ess
            consensus_vars['ess']['dso']['current'][node_id]['q'] = q_ess

        dso_models[node_id] = dso_model

    return dso_models, results


def create_admm_variables(operational_planning):

    consensus_variables = {
        'v_sqr': {'tso': {'current': dict(), 'prev': dict()},
                  'dso': {'current': dict(), 'prev': dict()}},
        'pf': {'tso': {'current': dict(), 'prev': dict()},
               'dso': {'current': dict(), 'prev': dict()}},
        'ess': {'tso': {'current': dict(), 'prev': dict()},
                'dso': {'current': dict(), 'prev': dict()},
                'esso': {'current': dict(), 'prev': dict()}}
    }

    dual_variables = {
        'v_sqr': {'tso': {'current': dict()}, 'dso': {'current': dict()}},
        'pf': {'tso': {'current': dict()}, 'dso': {'current': dict()}},
        'ess': {'tso': {'current': dict()}, 'dso': {'current': dict()}, 'esso': {'current': dict()}}
    }

    if operational_planning.params.previous_iter['ess']:
        dual_variables['ess']['tso']['prev'] = dict()
        dual_variables['ess']['dso']['prev'] = dict()

    for dn in range(len(operational_planning.active_distribution_network_nodes)):

        node_id = operational_planning.active_distribution_network_nodes[dn]
        node_base_kv = operational_planning.transmission_network.get_node_base_kv(node_id)

        consensus_variables['v_sqr']['tso']['current'][node_id] = node_base_kv
        consensus_variables['v_sqr']['dso']['current'][node_id] = node_base_kv
        consensus_variables['pf']['tso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['pf']['dso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['ess']['tso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['ess']['dso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['ess']['esso']['current'][node_id] = {'p': 0.00, 'q': 0.00}

        consensus_variables['v_sqr']['tso']['prev'][node_id] = node_base_kv
        consensus_variables['v_sqr']['dso']['prev'][node_id] = node_base_kv
        consensus_variables['pf']['tso']['prev'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['pf']['dso']['prev'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['ess']['tso']['prev'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['ess']['dso']['prev'][node_id] = {'p': 0.00, 'q': 0.00}
        consensus_variables['ess']['esso']['prev'][node_id] = {'p': 0.00, 'q': 0.00}

        dual_variables['v_sqr']['tso']['current'][node_id] = 0.00
        dual_variables['v_sqr']['dso']['current'][node_id] = 0.00
        dual_variables['pf']['tso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        dual_variables['pf']['dso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        dual_variables['ess']['tso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        dual_variables['ess']['dso']['current'][node_id] = {'p': 0.00, 'q': 0.00}
        dual_variables['ess']['esso']['current'][node_id] = {'p': 0.00, 'q': 0.00}

        if operational_planning.params.previous_iter['ess']:
            dual_variables['ess']['tso']['prev'][node_id] = {'p': 0.00, 'q': 0.00}
            dual_variables['ess']['dso']['prev'][node_id] = {'p': 0.00, 'q': 0.00}

    return consensus_variables, dual_variables


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
