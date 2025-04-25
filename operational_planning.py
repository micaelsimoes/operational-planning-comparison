import os
import time
import pandas as pd
from math import isclose, sqrt
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

    def run_distributed_coordination(self, t=0, consider_shared_ess=False, filename=str(), debug_flag=False):
        convergence, results, models, primal_evolution = _run_distributed_coordination(self, t, consider_shared_ess, debug_flag=debug_flag)
        if not filename:
            filename = self.name
        self.write_operational_planning_results_to_excel(models, results, filename=filename, primal_evolution=primal_evolution)
        return convergence, results, models, primal_evolution

    def write_operational_planning_results_to_excel(self, optimization_models, results, filename=str(), primal_evolution=list()):
        if not filename:
            filename = 'operational_planning_results'
        processed_results = _process_operational_planning_results(self, optimization_models['tso'], optimization_models['dso'], optimization_models['esso'], results)
        shared_ess_capacity = self.shared_ess_data.get_available_capacity(optimization_models['esso'])
        _write_operational_planning_results_to_excel(self, processed_results, primal_evolution=primal_evolution, shared_ess_capacity=shared_ess_capacity, filename=filename)

    def get_primal_value(self, tso_model, dso_models):
        return _get_primal_value(self, tso_model, dso_models)

    def update_admm_consensus_variables(self, tso_model, dso_models, consensus_vars, dual_vars, results, params, consider_shared_ess=False, update_tn=False, update_dns=False):
        self.update_interface_power_flow_variables(tso_model, dso_models, consensus_vars, dual_vars, results, params, update_tn=update_tn, update_dns=update_dns)
        if consider_shared_ess:
            self.update_shared_energy_storage_variables(tso_model, dso_models, consensus_vars['ess'], dual_vars['ess'], results, params, update_tn=update_tn, update_dns=update_dns)

    def update_interface_power_flow_variables(self, tso_model, dso_models, interface_vars, dual_vars, results, params, update_tn=True, update_dns=True):
        _update_interface_power_flow_variables(self, tso_model, dso_models, interface_vars, dual_vars, results, params, update_tn=update_tn, update_dns=update_dns)

    def update_shared_energy_storage_variables(self, tso_model, dso_models, consensus_vars, dual_vars, results, params, update_tn=True, update_dns=True):
        _update_shared_energy_storage_variables(self, tso_model, dso_models, consensus_vars, dual_vars, results, params, update_tn=update_tn, update_dns=update_dns)

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
def _run_distributed_coordination(operational_planning, t, consider_shared_ess, debug_flag=False):

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

    # Update models to ADMM
    update_distribution_models_to_admm(operational_planning, dso_models, admm_parameters, consider_shared_ess)
    update_transmission_model_to_admm(operational_planning, tso_model, admm_parameters, consider_shared_ess)

    # Update consensus variables
    operational_planning.update_admm_consensus_variables(tso_model, dso_models, consensus_vars, dual_vars, results, admm_parameters, consider_shared_ess, update_tn=True, update_dns=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ADMM -- Main cycle
    # ------------------------------------------------------------------------------------------------------------------
    convergence, iter = False, 1
    for iter in range(iter, admm_parameters.num_max_iters + 1):

        print(f'[INFO]\t - ADMM. Iter {iter}...')

        iter_start = time.time()

        # --------------------------------------------------------------------------------------------------------------
        # 1. Solve DSOs problems
        results['dso'] = update_distribution_coordination_models_and_solve(distribution_networks, dso_models,
                                                                           consensus_vars['v_sqr'], dual_vars['v_sqr']['dso'],
                                                                           consensus_vars['pf'], dual_vars['pf']['dso'],
                                                                           consensus_vars['ess'], dual_vars['ess']['dso'],
                                                                           admm_parameters, consider_shared_ess,
                                                                           from_warm_start=from_warm_start)

        # 1.1 Update ADMM CONSENSUS variables
        operational_planning.update_admm_consensus_variables(tso_model, dso_models,
                                                             consensus_vars, dual_vars, results, admm_parameters,
                                                             consider_shared_ess=consider_shared_ess, update_dns=True)

        # 1.2 Update primal evolution
        primal_evolution.append(operational_planning.get_primal_value(tso_model, dso_models))

        # 1.3 STOPPING CRITERIA evaluation
        if iter > 1:
            convergence = check_admm_convergence(operational_planning, consensus_vars, admm_parameters, consider_shared_ess, debug_flag=debug_flag)
            if convergence:
                iter_end = time.time()
                print('[INFO] \t - Iter {}: {:.2f} s'.format(iter, iter_end - iter_start))
                break

        # --------------------------------------------------------------------------------------------------------------
        # 2. Solve TSO problem
        results['tso'] = update_transmission_coordination_model_and_solve(transmission_network, tso_model,
                                                                          consensus_vars['v_sqr'], dual_vars['v_sqr']['tso'],
                                                                          consensus_vars['pf'], dual_vars['pf']['tso'],
                                                                          consensus_vars['ess'], dual_vars['ess']['tso'],
                                                                          admm_parameters, consider_shared_ess,
                                                                          from_warm_start=from_warm_start)

        # 2.1 Update ADMM CONSENSUS variables
        operational_planning.update_admm_consensus_variables(tso_model, dso_models,
                                                             consensus_vars, dual_vars, results, admm_parameters,
                                                             consider_shared_ess=consider_shared_ess, update_tn=True)

        # 2.2 Update primal evolution
        primal_evolution.append(operational_planning.get_primal_value(tso_model, dso_models))

        # 2.3 STOPPING CRITERIA evaluation
        convergence = check_admm_convergence(operational_planning, consensus_vars, admm_parameters, consider_shared_ess, debug_flag=debug_flag)
        if convergence:
            iter_end = time.time()
            print('[INFO] \t - Iter {}: {:.2f} s'.format(iter, iter_end - iter_start))
            break

        iter_end = time.time()
        print('[INFO] \t - Iter {}: {:.2f} s'.format(iter, iter_end - iter_start))

        from_warm_start = True

    if not convergence:
        print(f'[WARNING] ADMM did NOT converge in {admm_parameters.num_max_iters} iterations!')
    else:
        print(f'[INFO] \t - ADMM converged in {iter} iterations.')

    end = time.time()
    total_execution_time = end - start
    print('[INFO] \t - Execution time: {:.2f} s'.format(total_execution_time))

    optim_models = {'tso': tso_model, 'dso': dso_models}

    return convergence, results, optim_models, primal_evolution


def create_transmission_network_model(transmission_network, consensus_vars, t, consider_shared_ess):

    # Build model
    tso_model = transmission_network.build_model(t)
    s_base = transmission_network.baseMVA

    # Update model with expected interface values
    tso_model.active_distribution_networks = range(len(transmission_network.active_distribution_network_nodes))

    # Free Vmag, Pc, Qc at the interface nodes
    for dn in tso_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_node_idx = transmission_network.get_node_idx(adn_node_id)
        adn_load_idx = transmission_network.get_adn_load_idx(adn_node_id)
        _, v_max = transmission_network.get_node_voltage_limits(adn_node_id)
        for s_o in tso_model.scenarios_operation:

            tso_model.e[adn_node_idx, s_o].fixed = False
            tso_model.e[adn_node_idx, s_o].setub(v_max + SMALL_TOLERANCE)
            tso_model.e[adn_node_idx, s_o].setlb(-v_max - SMALL_TOLERANCE)
            tso_model.f[adn_node_idx, s_o].fixed = False
            tso_model.f[adn_node_idx, s_o].setub(v_max + SMALL_TOLERANCE)
            tso_model.f[adn_node_idx, s_o].setlb(-v_max - SMALL_TOLERANCE)
            if transmission_network.params.slacks.grid_operation.voltage:
                tso_model.slack_e[adn_node_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.slack_e[adn_node_idx, s_o].setlb(-EQUALITY_TOLERANCE)
                tso_model.slack_f[adn_node_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.slack_f[adn_node_idx, s_o].setlb(-EQUALITY_TOLERANCE)

            tso_model.pc[adn_load_idx, s_o].fixed = False
            tso_model.pc[adn_load_idx, s_o].setub(None)
            tso_model.pc[adn_load_idx, s_o].setlb(None)
            tso_model.qc[adn_load_idx, s_o].fixed = False
            tso_model.qc[adn_load_idx, s_o].setub(None)
            tso_model.qc[adn_load_idx, s_o].setlb(None)
            if transmission_network.params.fl_reg:
                tso_model.flex_p_up[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.flex_p_down[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.flex_q_up[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.flex_q_down[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
            if transmission_network.params.l_curt:
                tso_model.pc_curt_down[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.pc_curt_up[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.qc_curt_down[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)
                tso_model.qc_curt_up[adn_load_idx, s_o].setub(EQUALITY_TOLERANCE)

    # Add expected interface and shared ESS values
    tso_model.interface_expected_values = pe.ConstraintList()
    tso_model.expected_interface_vmag_sqr = pe.Var(tso_model.active_distribution_networks, domain=pe.NonNegativeReals, initialize=1.00)
    tso_model.expected_interface_pf_p = pe.Var(tso_model.active_distribution_networks, domain=pe.Reals, initialize=0.00)
    tso_model.expected_interface_pf_q = pe.Var(tso_model.active_distribution_networks, domain=pe.Reals, initialize=0.00)
    for dn in tso_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_node_idx = transmission_network.get_node_idx(adn_node_id)
        adn_load_idx = transmission_network.get_adn_load_idx(adn_node_id)
        expected_vmag_sqr = 0.00
        expected_pf_p = 0.00
        expected_pf_q = 0.00
        for s_o in tso_model.scenarios_operation:
            omega_oper = transmission_network.prob_operation_scenarios[s_o]
            expected_vmag_sqr += omega_oper * (tso_model.e[adn_node_idx, s_o] ** 2 + tso_model.f[adn_node_idx, s_o] ** 2)
            expected_pf_p += omega_oper * tso_model.pc[adn_load_idx, s_o]
            expected_pf_q += omega_oper * tso_model.qc[adn_load_idx, s_o]
        tso_model.interface_expected_values.add(tso_model.expected_interface_vmag_sqr[dn] <= expected_vmag_sqr + SMALL_TOLERANCE)
        tso_model.interface_expected_values.add(tso_model.expected_interface_vmag_sqr[dn] >= expected_vmag_sqr - SMALL_TOLERANCE)
        tso_model.interface_expected_values.add(tso_model.expected_interface_pf_p[dn] <= expected_pf_p + SMALL_TOLERANCE)
        tso_model.interface_expected_values.add(tso_model.expected_interface_pf_p[dn] >= expected_pf_p - SMALL_TOLERANCE)
        tso_model.interface_expected_values.add(tso_model.expected_interface_pf_q[dn] <= expected_pf_q + SMALL_TOLERANCE)
        tso_model.interface_expected_values.add(tso_model.expected_interface_pf_q[dn] >= expected_pf_q - SMALL_TOLERANCE)
    if consider_shared_ess:
        tso_model.expected_shared_ess_p = pe.Var(tso_model.shared_energy_storages, domain=pe.Reals, initialize=0.00)
        tso_model.expected_shared_ess_q = pe.Var(tso_model.shared_energy_storages, domain=pe.Reals, initialize=0.00)
        for e in tso_model.shared_energy_storages:
            expected_ess_p = 0.00
            expected_ess_q = 0.00
            for s_o in tso_model.scenarios_operation:
                omega_oper = transmission_network.prob_operation_scenarios[s_o]
                expected_ess_p += omega_oper * tso_model.shared_es_pnet[e, s_o]
                expected_ess_q += omega_oper * tso_model.shared_es_qnet[e, s_o]
            tso_model.interface_expected_values.add(tso_model.expected_shared_ess_p[e] <= expected_ess_p + SMALL_TOLERANCE)
            tso_model.interface_expected_values.add(tso_model.expected_shared_ess_p[e] >= expected_ess_p - SMALL_TOLERANCE)
            tso_model.interface_expected_values.add(tso_model.expected_shared_ess_q[e] <= expected_ess_q + SMALL_TOLERANCE)
            tso_model.interface_expected_values.add(tso_model.expected_shared_ess_q[e] >= expected_ess_q - SMALL_TOLERANCE)

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    obj = copy(tso_model.objective.expr)
    tso_model.penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
    tso_model.penalty_regularization.fix(PENALTY_REGULARIZATION)
    for dn in tso_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        adn_node_idx = transmission_network.get_node_idx(adn_node_id)
        adn_load_idx = transmission_network.get_adn_load_idx(adn_node_id)
        for s_o in tso_model.scenarios_operation:
            obj += tso_model.penalty_regularization * ((tso_model.e[adn_node_idx, s_o] ** 2 + tso_model.f[adn_node_idx, s_o] ** 2) - tso_model.expected_interface_vmag_sqr[dn]) ** 2
            obj += tso_model.penalty_regularization * s_base * (tso_model.pc[adn_load_idx, s_o] - tso_model.expected_interface_pf_p[dn]) ** 2
            obj += tso_model.penalty_regularization * s_base * (tso_model.qc[adn_load_idx, s_o] - tso_model.expected_interface_pf_q[dn]) ** 2
    if consider_shared_ess:
        for e in tso_model.shared_energy_storages:
            for s_o in tso_model.scenarios_operation:
                obj += tso_model.penalty_regularization * s_base * (tso_model.shared_es_pnet[e, s_o] - tso_model.expected_shared_ess_p[e]) ** 2
                obj += tso_model.penalty_regularization * s_base * (tso_model.shared_es_qnet[e, s_o] - tso_model.expected_shared_ess_q[e]) ** 2
    tso_model.objective.expr = obj

    # Fix initial values, run OPF
    for dn in tso_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        v_base = transmission_network.get_node_base_kv(adn_node_id)
        interface_v_sqr = consensus_vars['v_sqr']['dso']['current'][adn_node_id] / (v_base ** 2)
        interface_pf_p = consensus_vars['pf']['dso']['current'][adn_node_id]['p'] / s_base
        interface_pf_q = consensus_vars['pf']['dso']['current'][adn_node_id]['q'] / s_base
        tso_model.expected_interface_vmag_sqr[dn].fix(interface_v_sqr)
        tso_model.expected_interface_pf_p[dn].fix(interface_pf_p)
        tso_model.expected_interface_pf_q[dn].fix(interface_pf_q)

    # Run SMOPF
    results = transmission_network.optimize(tso_model)

    # Get initial interface and shared ESS values
    for dn in tso_model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        v_base = transmission_network.get_node_base_kv(adn_node_id)
        interface_v_sqr = pe.value(tso_model.expected_interface_vmag_sqr[dn]) * (v_base ** 2)
        interface_pf_p = pe.value(tso_model.expected_interface_pf_p[dn]) * s_base
        interface_pf_q = pe.value(tso_model.expected_interface_pf_q[dn]) * s_base
        consensus_vars['v_sqr']['tso']['current'][adn_node_id] = interface_v_sqr
        consensus_vars['pf']['tso']['current'][adn_node_id]['p'] = interface_pf_p
        consensus_vars['pf']['tso']['current'][adn_node_id]['q'] = interface_pf_q
        if consider_shared_ess:
            shared_ess_idx = transmission_network.get_shared_energy_storage_idx(adn_node_id)
            p_ess = pe.value(tso_model.expected_shared_ess_p[shared_ess_idx]) * s_base
            q_ess = pe.value(tso_model.expected_shared_ess_q[shared_ess_idx]) * s_base
            consensus_vars['ess']['tso']['current'][adn_node_id]['p'] = p_ess
            consensus_vars['ess']['tso']['current'][adn_node_id]['q'] = q_ess

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


def update_transmission_model_to_admm(operational_planning, model, params, consider_shared_ess):

    transmission_network = operational_planning.transmission_network
    distribution_networks = operational_planning.distribution_networks

    s_base = transmission_network.baseMVA

    # Free expected values
    for dn in model.active_distribution_networks:
        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        v_min, v_max = transmission_network.get_node_voltage_limits(adn_node_id)
        model.expected_interface_vmag_sqr[dn].fixed = False
        model.expected_interface_vmag_sqr[dn].setub(v_max ** 2 + SMALL_TOLERANCE)
        model.expected_interface_vmag_sqr[dn].setlb(v_min ** 2 - SMALL_TOLERANCE)
        model.expected_interface_pf_p[dn].fixed = False
        model.expected_interface_pf_p[dn].setub(None)
        model.expected_interface_pf_p[dn].setlb(None)
        model.expected_interface_pf_q[dn].fixed = False
        model.expected_interface_pf_q[dn].setub(None)
        model.expected_interface_pf_q[dn].setlb(None)
        if consider_shared_ess:
            shared_ess_idx = transmission_network.get_shared_energy_storage_idx(adn_node_id)
            model.expected_shared_ess_p[shared_ess_idx].setub(None)
            model.expected_shared_ess_p[shared_ess_idx].setlb(None)
            model.expected_shared_ess_q[shared_ess_idx].setub(None)
            model.expected_shared_ess_q[shared_ess_idx].setlb(None)

    # Update costs (penalties) for the coordination procedure
    model.penalty_ess_usage.fix(1e-6)
    if transmission_network.params.obj_type == OBJ_MIN_COST:
        model.cost_res_curtailment.fix(COST_GENERATION_CURTAILMENT)
        model.cost_load_curtailment.fix(COST_CONSUMPTION_CURTAILMENT)
    elif transmission_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        model.penalty_gen_curtailment.fix(1e-2)
        model.penalty_load_curtailment.fix(PENALTY_LOAD_CURTAILMENT)
        model.penalty_flex_usage.fix(1e-3)

    # Add ADMM variables
    model.rho_v = pe.Var(domain=pe.NonNegativeReals)
    model.rho_v.fix(params.rho['v'][transmission_network.name])
    model.v_sqr_req = pe.Var(model.active_distribution_networks, domain=pe.NonNegativeReals)    # Square of voltage magnitude
    model.dual_v_sqr_req = pe.Var(model.active_distribution_networks, domain=pe.Reals)          # Dual variable - voltage magnitude requested

    model.rho_pf = pe.Var(domain=pe.NonNegativeReals)
    model.rho_pf.fix(params.rho['pf'][transmission_network.name])
    model.p_pf_req = pe.Var(model.active_distribution_networks, domain=pe.Reals)                # Active power - requested by distribution networks
    model.q_pf_req = pe.Var(model.active_distribution_networks, domain=pe.Reals)                # Reactive power - requested by distribution networks
    model.dual_pf_p_req = pe.Var(model.active_distribution_networks, domain=pe.Reals)           # Dual variable - active power requested
    model.dual_pf_q_req = pe.Var(model.active_distribution_networks, domain=pe.Reals)           # Dual variable - reactive power requested

    if consider_shared_ess:
        model.rho_ess = pe.Var(domain=pe.NonNegativeReals)
        model.rho_ess.fix(params.rho['ess'][transmission_network.name])
        model.p_ess_req = pe.Var(model.shared_energy_storages, domain=pe.Reals)                     # Shared ESS - Active power requested (DSO)
        model.q_ess_req = pe.Var(model.shared_energy_storages, domain=pe.Reals)                     # Shared ESS - Reactive power requested (DSO)
        model.dual_ess_p_req = pe.Var(model.shared_energy_storages, domain=pe.Reals)                # Dual variable - Shared ESS active power
        model.dual_ess_q_req = pe.Var(model.shared_energy_storages, domain=pe.Reals)                # Dual variable - Shared ESS reactive power
        if params.previous_iter['ess']['tso']:
            model.rho_ess_prev = pe.Var(domain=pe.NonNegativeReals)
            model.rho_ess_prev.fix(params.rho_previous_iter['ess'][transmission_network.name])
            model.p_ess_prev = pe.Var(model.shared_energy_storages, domain=pe.Reals)                # Shared ESS - previous iteration active power
            model.q_ess_prev = pe.Var(model.shared_energy_storages, domain=pe.Reals)                # Shared ESS - previous iteration reactive power
            model.dual_ess_p_prev = pe.Var(model.shared_energy_storages, domain=pe.Reals)           # Dual variable - previous iteration shared ESS active power
            model.dual_ess_q_prev = pe.Var(model.shared_energy_storages, domain=pe.Reals)           # Dual variable - previous iteration shared ESS reactive power

    # Objective function - augmented Lagrangian
    init_of_value = 1.00
    if transmission_network.params.obj_type == OBJ_MIN_COST:
        init_of_value = abs(pe.value(model.objective))
    if isclose(init_of_value, 0.00, abs_tol=SMALL_TOLERANCE):
        init_of_value = 0.01
    obj = copy(model.objective.expr) / init_of_value

    for dn in model.active_distribution_networks:

        adn_node_id = transmission_network.active_distribution_network_nodes[dn]
        distribution_network = distribution_networks[adn_node_id]
        interface_transf_rating = distribution_network.get_interface_branch_rating() / s_base

        constraint_v_req = (model.expected_interface_vmag_sqr[dn] - model.v_sqr_req[dn])
        obj += model.dual_v_sqr_req[dn] * constraint_v_req
        obj += (model.rho_v / 2) * (constraint_v_req ** 2)

        constraint_p_req = (model.expected_interface_pf_p[dn] - model.p_pf_req[dn]) / interface_transf_rating
        constraint_q_req = (model.expected_interface_pf_q[dn] - model.q_pf_req[dn]) / interface_transf_rating
        obj += model.dual_pf_p_req[dn] * constraint_p_req
        obj += model.dual_pf_q_req[dn] * constraint_q_req
        obj += (model.rho_pf / 2) * (constraint_p_req ** 2)
        obj += (model.rho_pf / 2) * (constraint_q_req ** 2)

    if consider_shared_ess:
        for e in model.shared_energy_storages:

            shared_ess_rating = abs(transmission_network.network.shared_energy_storages[e].s)
            if isclose(shared_ess_rating, 0.00, abs_tol=SMALL_TOLERANCE):
                shared_ess_rating = 0.01

            constraint_ess_p_req = (model.expected_shared_ess_p[e] - model.p_ess_req[e]) / (2 * shared_ess_rating)
            constraint_ess_q_req = (model.expected_shared_ess_q[e] - model.q_ess_req[e]) / (2 * shared_ess_rating)
            obj += (model.dual_ess_p_req[e]) * constraint_ess_p_req
            obj += (model.dual_ess_q_req[e]) * constraint_ess_q_req
            obj += (model.rho_ess / 2) * constraint_ess_p_req ** 2
            obj += (model.rho_ess / 2) * constraint_ess_q_req ** 2
            if params.previous_iter['ess']['tso']:
                constraint_ess_p_prev = (model.expected_shared_ess_p[e] - model.p_ess_prev[e]) / (2 * shared_ess_rating)
                constraint_ess_q_prev = (model.expected_shared_ess_q[e] - model.q_ess_prev[e]) / (2 * shared_ess_rating)
                obj += (model.rho_ess_prev / 2) * constraint_ess_p_prev ** 2
                obj += (model.rho_ess_prev / 2) * constraint_ess_q_prev ** 2

    # Add ADMM OF, deactivate original OF
    model.objective.deactivate()
    model.admm_objective = pe.Objective(sense=pe.minimize, expr=obj)


def update_distribution_models_to_admm(operational_planning, models, params, consider_shared_ess):

    distribution_networks = operational_planning.distribution_networks

    for node_id in distribution_networks:

        dso_model = models[node_id]
        distribution_network = distribution_networks[node_id]
        s_base = distribution_network.baseMVA
        ref_node_id = distribution_network.get_reference_node_id()
        ref_node_idx = distribution_network.get_node_idx(ref_node_id)
        ref_gen_idx = distribution_network.get_reference_gen_idx()
        _, v_max = distribution_network.get_node_voltage_limits(ref_node_id)

        # Update Vmag, Pg, Qg limits at the interface node
        for s_o in dso_model.scenarios_operation:
            dso_model.e[ref_node_idx, s_o].fixed = False
            dso_model.e[ref_node_idx, s_o].setub(v_max + SMALL_TOLERANCE)
            dso_model.e[ref_node_idx, s_o].setlb(-v_max - SMALL_TOLERANCE)
            dso_model.f[ref_node_idx, s_o].setub(SMALL_TOLERANCE)
            dso_model.f[ref_node_idx, s_o].setlb(-SMALL_TOLERANCE)
            if distribution_network.params.slacks.grid_operation.voltage:
                dso_model.slack_e[ref_node_idx, s_o].setub(SMALL_TOLERANCE)
                dso_model.slack_e[ref_node_idx, s_o].setlb(-SMALL_TOLERANCE)
                dso_model.slack_f[ref_node_idx, s_o].setub(SMALL_TOLERANCE)
                dso_model.slack_f[ref_node_idx, s_o].setlb(-SMALL_TOLERANCE)
            dso_model.pg[ref_gen_idx, s_o].fixed = False
            dso_model.qg[ref_gen_idx, s_o].fixed = False
            if distribution_network.params.rg_curt:
                dso_model.sg_curt[ref_gen_idx, s_o].setub(SMALL_TOLERANCE)

        # Update expected interface values limits
        dso_model.expected_interface_vmag_sqr.fixed = False
        dso_model.expected_interface_vmag_sqr.setub(None)
        dso_model.expected_interface_vmag_sqr.setlb(None)
        dso_model.expected_interface_pf_p.fixed = False
        dso_model.expected_interface_pf_p.setub(None)
        dso_model.expected_interface_pf_p.setlb(None)
        dso_model.expected_interface_pf_q.fixed = False
        dso_model.expected_interface_pf_q.setub(None)
        dso_model.expected_interface_pf_q.setlb(None)
        if consider_shared_ess:
            dso_model.expected_shared_ess_p.fixed = False
            dso_model.expected_shared_ess_p.setub(None)
            dso_model.expected_shared_ess_p.setlb(None)
            dso_model.expected_shared_ess_q.fixed = False
            dso_model.expected_shared_ess_q.setub(None)
            dso_model.expected_shared_ess_q.setlb(None)

        # Update costs (penalties) for the coordination procedure
        dso_model.penalty_ess_usage.fix(0.00)
        if distribution_network.params.obj_type == OBJ_MIN_COST:
            dso_model.cost_res_curtailment.fix(0.00)
            dso_model.cost_load_curtailment.fix(COST_CONSUMPTION_CURTAILMENT)
        elif distribution_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
            dso_model.penalty_gen_curtailment.fix(0.00)
            dso_model.penalty_load_curtailment.fix(PENALTY_LOAD_CURTAILMENT)
            dso_model.penalty_flex_usage.fix(0.00)

        # Add ADMM variables
        dso_model.rho_v = pe.Var(domain=pe.NonNegativeReals)
        dso_model.rho_v.fix(params.rho['v'][distribution_network.name])
        dso_model.v_sqr_req = pe.Var(domain=pe.NonNegativeReals)       # Voltage magnitude - requested by TSO
        dso_model.dual_v_sqr_req = pe.Var(domain=pe.Reals)             # Dual variable - voltage magnitude

        dso_model.rho_pf = pe.Var(domain=pe.NonNegativeReals)
        dso_model.rho_pf.fix(params.rho['pf'][distribution_network.name])
        dso_model.p_pf_req = pe.Var(domain=pe.Reals)                   # Active power - requested by TSO
        dso_model.q_pf_req = pe.Var(domain=pe.Reals)                   # Reactive power - requested by TSO
        dso_model.dual_pf_p_req = pe.Var(domain=pe.Reals)              # Dual variable - active power
        dso_model.dual_pf_q_req = pe.Var(domain=pe.Reals)              # Dual variable - reactive power

        if consider_shared_ess:
            dso_model.rho_ess = pe.Var(domain=pe.NonNegativeReals)
            dso_model.rho_ess.fix(params.rho['ess'][distribution_network.name])
            dso_model.p_ess_req = pe.Var(domain=pe.Reals)                  # Shared ESS - active power requested (TSO)
            dso_model.q_ess_req = pe.Var(domain=pe.Reals)                  # Shared ESS - reactive power requested (TSO)
            dso_model.dual_ess_p_req = pe.Var(domain=pe.Reals)             # Dual variable - Shared ESS active power
            dso_model.dual_ess_q_req = pe.Var(domain=pe.Reals)             # Dual variable - Shared ESS reactive power
            if params.previous_iter['ess']['dso']:
                dso_model.rho_ess_prev = pe.Var(domain=pe.NonNegativeReals)
                dso_model.rho_ess_prev.fix(params.rho_previous_iter['ess'][distribution_network.name])
                dso_model.p_ess_prev = pe.Var(domain=pe.Reals)             # Shared ESS - previous iteration active power
                dso_model.q_ess_prev = pe.Var(domain=pe.Reals)             # Shared ESS - previous iteration reactive power
                dso_model.dual_ess_p_prev = pe.Var(domain=pe.Reals)        # Dual variable - Shared ESS previous iteration active power
                dso_model.dual_ess_q_prev = pe.Var(domain=pe.Reals)        # Dual variable - Shared ESS previous iteration reactive power

        # Objective function - augmented Lagrangian
        init_of_value = 1.00
        if distribution_network.params.obj_type == OBJ_MIN_COST:
            init_of_value = abs(pe.value(dso_model.objective))
        if isclose(init_of_value, 0.00, abs_tol=SMALL_TOLERANCE):
            init_of_value = 0.01
        obj = copy(dso_model.objective.expr) / init_of_value

        if consider_shared_ess:
            shared_ess_idx = distribution_network.get_shared_energy_storage_idx(ref_node_id)
            shared_ess_rating = abs(distribution_network.shared_energy_storages[shared_ess_idx].s)
            if isclose(shared_ess_rating, 0.00, abs_tol=SMALL_TOLERANCE):
                shared_ess_rating = 0.01

        interface_transf_rating = distribution_network.get_interface_branch_rating() / s_base

        # Augmented Lagrangian -- Interface power flow (residual balancing)

        # Voltage magnitude
        constraint_vmag_req = (dso_model.expected_interface_vmag_sqr - dso_model.v_sqr_req)
        obj += (dso_model.dual_v_sqr_req) * constraint_vmag_req
        obj += (dso_model.rho_v / 2) * (constraint_vmag_req ** 2)

        # Interface power flow
        constraint_p_req = (dso_model.expected_interface_pf_p - dso_model.p_pf_req) / interface_transf_rating
        constraint_q_req = (dso_model.expected_interface_pf_q - dso_model.q_pf_req) / interface_transf_rating
        obj += (dso_model.dual_pf_p_req) * constraint_p_req
        obj += (dso_model.dual_pf_q_req) * constraint_q_req
        obj += (dso_model.rho_pf / 2) * (constraint_p_req ** 2)
        obj += (dso_model.rho_pf / 2) * (constraint_q_req ** 2)

        # Shared ESS
        if consider_shared_ess:
            constraint_ess_p_req = (dso_model.expected_shared_ess_p - dso_model.p_ess_req) / (2 * shared_ess_rating)
            constraint_ess_q_req = (dso_model.expected_shared_ess_q - dso_model.q_ess_req) / (2 * shared_ess_rating)
            obj += (dso_model.dual_ess_p_req) * constraint_ess_p_req
            obj += (dso_model.dual_ess_q_req) * constraint_ess_q_req
            obj += (dso_model.rho_ess / 2) * constraint_ess_p_req ** 2
            obj += (dso_model.rho_ess / 2) * constraint_ess_q_req ** 2
            if params.previous_iter['ess']['dso']:
                constraint_ess_p_prev = (dso_model.expected_shared_ess_p - dso_model.p_ess_prev) / (2 * shared_ess_rating)
                constraint_ess_q_prev = (dso_model.expected_shared_ess_q - dso_model.q_ess_prev) / (2 * shared_ess_rating)
                obj += (dso_model.dual_ess_p_prev) * constraint_ess_p_prev
                obj += (dso_model.dual_ess_q_prev) * constraint_ess_q_prev
                obj += (dso_model.rho_ess_prev / 2) * constraint_ess_p_prev ** 2
                obj += (dso_model.rho_ess_prev / 2) * constraint_ess_q_prev ** 2

        # Add ADMM OF, deactivate original OF
        dso_model.objective.deactivate()
        dso_model.admm_objective = pe.Objective(sense=pe.minimize, expr=obj)


def update_transmission_coordination_model_and_solve(transmission_network, model, vsqr_req, dual_vsqr, pf_req, dual_pf, ess_req, dual_ess, params, consider_shared_ess, from_warm_start=False):

    print('[INFO] \t\t - Updating transmission network...')

    s_base = transmission_network.baseMVA

    rho_v = params.rho['v'][transmission_network.name]
    rho_pf = params.rho['pf'][transmission_network.name]
    if params.adaptive_penalty:
        rho_v = pe.value(model.rho_v) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
        rho_pf = pe.value(model.rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)

    if consider_shared_ess:
        rho_ess = params.rho['ess'][transmission_network.name]
        if params.adaptive_penalty:
            rho_ess = pe.value(model.rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
        if params.previous_iter['ess']['tso']:
            rho_ess_prev = params.rho_previous_iter['ess'][transmission_network.name]
            if params.adaptive_penalty:
                rho_ess_prev = pe.value(model.rho_ess_prev) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)

    # Update Rho parameter
    model.rho_v.fix(rho_v)
    model.rho_pf.fix(rho_pf)
    if consider_shared_ess:
        model.rho_ess.fix(rho_ess)
        if params.previous_iter['ess']['tso']:
            model.rho_ess_prev.fix(rho_ess_prev)

    for dn in model.active_distribution_networks:

        node_id = transmission_network.active_distribution_network_nodes[dn]
        v_base = transmission_network.get_node_base_kv(node_id)

        # Update VOLTAGE and POWER FLOW variables at connection point
        model.dual_v_sqr_req[dn].fix(dual_vsqr['current'][node_id] / (v_base ** 2))
        model.v_sqr_req[dn].fix(vsqr_req['dso']['current'][node_id] / (v_base ** 2))
        model.dual_pf_p_req[dn].fix(dual_pf['current'][node_id]['p'] / s_base)
        model.dual_pf_q_req[dn].fix(dual_pf['current'][node_id]['q'] / s_base)
        model.p_pf_req[dn].fix(pf_req['dso']['current'][node_id]['p'] / s_base)
        model.q_pf_req[dn].fix(pf_req['dso']['current'][node_id]['q'] / s_base)

        # Update shared ESS capacity and power requests
        if consider_shared_ess:
            shared_ess_idx = transmission_network.network.get_shared_energy_storage_idx(node_id)
            model.dual_ess_p_req[shared_ess_idx].fix(dual_ess['current'][node_id]['p'] / s_base)
            model.dual_ess_q_req[shared_ess_idx].fix(dual_ess['current'][node_id]['q'] / s_base)
            model.p_ess_req[shared_ess_idx].fix(ess_req['dso']['current'][node_id]['p'] / s_base)
            model.q_ess_req[shared_ess_idx].fix(ess_req['dso']['current'][node_id]['q'] / s_base)
            if params.previous_iter['ess']['tso']:
                model.dual_ess_p_prev[shared_ess_idx].fix(dual_ess['prev'][node_id]['p'] / s_base)
                model.dual_ess_q_prev[shared_ess_idx].fix(dual_ess['prev'][node_id]['q'] / s_base)
                model.p_ess_prev[shared_ess_idx].fix(ess_req['tso']['prev'][node_id]['p'] / s_base)
                model.q_ess_prev[shared_ess_idx].fix(ess_req['tso']['prev'][node_id]['q'] / s_base)

    # Solve!
    res = transmission_network.optimize(model, from_warm_start=from_warm_start)
    if res.solver.status == po.SolverStatus.error:
        print(f'[ERROR] Network {model.name} did not converge!')
        # exit(ERROR_NETWORK_OPTIMIZATION)

    return res


def update_distribution_coordination_models_and_solve(distribution_networks, models, vsqr_req, dual_vsqr, pf_req, dual_pf, ess_req, dual_ess, params, consider_shared_ess, from_warm_start=False):

    print('[INFO] \t\t - Updating distribution networks:')
    res = dict()

    for node_id in distribution_networks:

        model = models[node_id]
        distribution_network = distribution_networks[node_id]

        print(f'[INFO] \t\t\t - Updating active distribution network connected to node {node_id}...')

        ref_node_id = distribution_network.get_reference_node_id()
        v_base = distribution_network.get_node_base_kv(ref_node_id)
        s_base = distribution_network.baseMVA

        rho_v = params.rho['v'][distribution_network.name]
        rho_pf = params.rho['pf'][distribution_network.name]
        if consider_shared_ess:
            rho_ess = params.rho['ess'][distribution_network.name]
        if params.adaptive_penalty:
            rho_v = pe.value(model.rho_v) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
            rho_pf = pe.value(model.rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
            if consider_shared_ess:
                rho_ess = pe.value(model.rho_ess) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
                if params.previous_iter['ess']['dso']:
                    rho_ess_prev = params.rho_previous_iter['ess'][distribution_network.name]
                    if params.adaptive_penalty:
                        rho_ess_prev = pe.value(model.rho_ess_prev) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)

        # Update Rho parameter
        model.rho_v.fix(rho_v)
        model.rho_pf.fix(rho_pf)
        if consider_shared_ess:
            model.rho_ess.fix(rho_ess)
            if consider_shared_ess:
                if params.previous_iter['ess']['dso']:
                    model.rho_ess_prev.fix(rho_ess_prev)

        # Update VOLTAGE and POWER FLOW variables at connection point
        model.dual_v_sqr_req.fix(dual_vsqr['current'][node_id] / (v_base ** 2))
        model.v_sqr_req.fix(vsqr_req['tso']['current'][node_id] / (v_base ** 2))
        model.dual_pf_p_req.fix(dual_pf['current'][node_id]['p'] / s_base)
        model.dual_pf_q_req.fix(dual_pf['current'][node_id]['q'] / s_base)
        model.p_pf_req.fix(pf_req['tso']['current'][node_id]['p'] / s_base)
        model.q_pf_req.fix(pf_req['tso']['current'][node_id]['q'] / s_base)

        # Update SHARED ENERGY STORAGE variables (if existent)
        if consider_shared_ess:
            model.dual_ess_p_req.fix(dual_ess['current'][node_id]['p'] / s_base)
            model.dual_ess_q_req.fix(dual_ess['current'][node_id]['q'] / s_base)
            model.p_ess_req.fix(ess_req['esso']['current'][node_id]['p'] / s_base)
            model.q_ess_req.fix(ess_req['esso']['current'][node_id]['q'] / s_base)
            if params.previous_iter['ess']['dso']:
                model.dual_ess_p_prev.fix(dual_ess['prev'][node_id]['p'] / s_base)
                model.dual_ess_q_prev.fix(dual_ess['prev'][node_id]['q'] / s_base)
                model.p_ess_prev.fix(ess_req['dso']['prev'][node_id]['p'] / s_base)
                model.q_ess_prev.fix(ess_req['dso']['prev'][node_id]['q'] / s_base)

        # Solve!
        res[node_id] = distribution_network.optimize(model, from_warm_start=from_warm_start)
        if res[node_id].solver.status != po.SolverStatus.ok:
            print(f'[WARNING] Network {model.name} did not converge!')
            #exit(ERROR_NETWORK_OPTIMIZATION)

    return res


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


def _update_interface_power_flow_variables(operational_planning, tso_model, dso_models, interface_vars, dual_vars, results, params, update_tn=True, update_dns=True):

    transmission_network = operational_planning.transmission_network
    distribution_networks = operational_planning.distribution_networks

    # Transmission network - Update Vmag and PF at the TN-DN interface
    if update_tn:
        for dn in range(len(operational_planning.active_distribution_network_nodes)):
            node_id = operational_planning.active_distribution_network_nodes[dn]
            v_base = transmission_network.get_node_base_kv(node_id)
            s_base = transmission_network.baseMVA
            if results['tso'].solver.status == po.SolverStatus.ok:
                interface_vars['v_sqr']['tso']['prev'][node_id] = copy(interface_vars['v_sqr']['tso']['current'][node_id])
                interface_vars['pf']['tso']['prev'][node_id]['p'] = copy(interface_vars['pf']['tso']['current'][node_id]['p'])
                interface_vars['pf']['tso']['prev'][node_id]['q'] = copy(interface_vars['pf']['tso']['current'][node_id]['q'])

                vsqr_req = pe.value(tso_model.expected_interface_vmag_sqr[dn]) * (v_base ** 2)
                p_req = pe.value(tso_model.expected_interface_pf_p[dn]) * s_base
                q_req = pe.value(tso_model.expected_interface_pf_q[dn]) * s_base
                interface_vars['v_sqr']['tso']['current'][node_id] = vsqr_req
                interface_vars['pf']['tso']['current'][node_id]['p'] = p_req
                interface_vars['pf']['tso']['current'][node_id]['q'] = q_req

    # Distribution Network - Update PF at the TN-DN interface
    if update_dns:
        for node_id in distribution_networks:
            distribution_network = distribution_networks[node_id]
            dso_model = dso_models[node_id]
            ref_node_id = distribution_network.get_reference_node_id()
            v_base = distribution_network.get_node_base_kv(ref_node_id)
            s_base = distribution_network.baseMVA
            if results['dso'][node_id].solver.status == po.SolverStatus.ok:
                interface_vars['v_sqr']['dso']['prev'][node_id] = copy(interface_vars['v_sqr']['dso']['current'][node_id])
                interface_vars['pf']['dso']['prev'][node_id]['p'] = copy(interface_vars['pf']['dso']['current'][node_id]['p'])
                interface_vars['pf']['dso']['prev'][node_id]['q'] = copy(interface_vars['pf']['dso']['current'][node_id]['q'])

                vsqr_req = pe.value(dso_model.expected_interface_vmag_sqr) * (v_base ** 2)
                p_req = pe.value(dso_model.expected_interface_pf_p) * s_base
                q_req = pe.value(dso_model.expected_interface_pf_q) * s_base
                interface_vars['v_sqr']['dso']['current'][node_id] = vsqr_req
                interface_vars['pf']['dso']['current'][node_id]['p'] = p_req
                interface_vars['pf']['dso']['current'][node_id]['q'] = q_req

    # Update Lambdas
    for node_id in distribution_networks:
        if update_tn:
            rho_v_tso = pe.value(tso_model.rho_v)
            rho_pf_tso = pe.value(tso_model.rho_pf)
            error_v_req_tso = interface_vars['v_sqr']['tso']['current'][node_id] - interface_vars['v_sqr']['dso']['current'][node_id]
            error_p_pf_req_tso = interface_vars['pf']['tso']['current'][node_id]['p'] - interface_vars['pf']['dso']['current'][node_id]['p']
            error_q_pf_req_tso = interface_vars['pf']['tso']['current'][node_id]['q'] - interface_vars['pf']['dso']['current'][node_id]['q']
            dual_vars['v_sqr']['tso']['current'][node_id] += rho_v_tso * error_v_req_tso
            dual_vars['pf']['tso']['current'][node_id]['p'] += rho_pf_tso * error_p_pf_req_tso
            dual_vars['pf']['tso']['current'][node_id]['q'] += rho_pf_tso * error_q_pf_req_tso

        if update_dns:
            rho_v_dso = pe.value(dso_models[node_id].rho_v)
            rho_pf_dso = pe.value(dso_models[node_id].rho_pf)
            error_v_req_dso = interface_vars['v_sqr']['dso']['current'][node_id] - interface_vars['v_sqr']['tso']['current'][node_id]
            error_p_pf_req_dso = interface_vars['pf']['dso']['current'][node_id]['p'] - interface_vars['pf']['tso']['current'][node_id]['p']
            error_q_pf_req_dso = interface_vars['pf']['dso']['current'][node_id]['q'] - interface_vars['pf']['tso']['current'][node_id]['q']
            dual_vars['v_sqr']['dso']['current'][node_id] += rho_v_dso * error_v_req_dso
            dual_vars['pf']['dso']['current'][node_id]['p'] += rho_pf_dso * error_p_pf_req_dso
            dual_vars['pf']['dso']['current'][node_id]['q'] += rho_pf_dso * error_q_pf_req_dso


def _update_shared_energy_storage_variables(operational_planning, tso_model, dso_models, shared_ess_vars, dual_vars, results, params, update_tn=True, update_dns=True):

    transmission_network = operational_planning.transmission_network
    distribution_networks = operational_planning.distribution_networks

    for node_id in operational_planning.active_distribution_network_nodes:

        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]

        # Power requested by TSO
        if update_tn:
            if results['tso'].solver.status == po.SolverStatus.ok:
                s_base = transmission_network.baseMVA
                shared_ess_idx = transmission_network.get_shared_energy_storage_idx(node_id)

                shared_ess_vars['tso']['prev'][node_id]['p'] = copy(shared_ess_vars['tso']['current'][node_id]['p'])
                shared_ess_vars['tso']['prev'][node_id]['q'] = copy(shared_ess_vars['tso']['current'][node_id]['q'])

                p_req = pe.value(tso_model.expected_shared_ess_p[shared_ess_idx]) * s_base
                q_req = pe.value(tso_model.expected_shared_ess_q[shared_ess_idx]) * s_base
                shared_ess_vars['tso']['current'][node_id]['p'] = p_req
                shared_ess_vars['tso']['current'][node_id]['q'] = q_req

        # Power requested by DSO
        if update_dns:
            if results['dso'][node_id].solver.status == po.SolverStatus.ok:
                s_base = distribution_network.baseMVA
                shared_ess_vars['dso']['prev'][node_id]['p'] = copy(shared_ess_vars['dso']['current'][node_id]['p'])
                shared_ess_vars['dso']['prev'][node_id]['q'] = copy(shared_ess_vars['dso']['current'][node_id]['q'])

                p_req = pe.value(dso_model.expected_shared_ess_p) * s_base
                q_req = pe.value(dso_model.expected_shared_ess_q) * s_base
                shared_ess_vars['dso']['current'][node_id]['p'] = p_req
                shared_ess_vars['dso']['current'][node_id]['q'] = q_req

        # Update dual variables Shared ESS
        if update_tn:
            rho_ess_tso = pe.value(tso_model.rho_ess)
            error_p_tso_dso = shared_ess_vars['tso']['current'][node_id]['p'] - shared_ess_vars['dso']['current'][node_id]['p']
            error_q_tso_dso = shared_ess_vars['tso']['current'][node_id]['q'] - shared_ess_vars['dso']['current'][node_id]['q']
            dual_vars['tso']['current'][node_id]['p'] += rho_ess_tso * error_p_tso_dso
            dual_vars['tso']['current'][node_id]['q'] += rho_ess_tso * error_q_tso_dso
            if params.previous_iter['ess']['tso']:
                error_p_tso_prev = shared_ess_vars['tso']['current'][node_id]['p'] - shared_ess_vars['tso']['prev'][node_id]['p']
                error_q_tso_prev = shared_ess_vars['tso']['current'][node_id]['q'] - shared_ess_vars['tso']['prev'][node_id]['q']
                dual_vars['tso']['prev'][node_id]['p'] += rho_ess_tso * error_p_tso_prev
                dual_vars['tso']['prev'][node_id]['q'] += rho_ess_tso * error_q_tso_prev

        if update_dns:
            rho_ess_dso = pe.value(dso_models[node_id].rho_ess)
            error_p_dso_tso = shared_ess_vars['dso']['current'][node_id]['p'] - shared_ess_vars['tso']['current'][node_id]['p']
            error_q_dso_tso = shared_ess_vars['dso']['current'][node_id]['q'] - shared_ess_vars['tso']['current'][node_id]['q']
            dual_vars['dso']['current'][node_id]['p'] += rho_ess_dso * error_p_dso_tso
            dual_vars['dso']['current'][node_id]['q'] += rho_ess_dso * error_q_dso_tso
            if params.previous_iter['ess']['dso']:
                error_p_dso_prev = shared_ess_vars['dso']['current'][node_id]['p'] - shared_ess_vars['dso']['prev'][node_id]['p']
                error_q_dso_prev = shared_ess_vars['dso']['current'][node_id]['q'] - shared_ess_vars['dso']['prev'][node_id]['q']
                dual_vars['dso']['prev'][node_id]['p'] += rho_ess_dso * error_p_dso_prev
                dual_vars['dso']['prev'][node_id]['q'] += rho_ess_dso * error_q_dso_prev


def _get_primal_value(operational_planning, tso_model, dso_models):

    transmission_network = operational_planning.transmission_network
    distribution_networks = operational_planning.distribution_networks

    primal_value = 0.0
    primal_value += transmission_network.get_primal_value(tso_model)
    for node_id in distribution_networks:
        primal_value += distribution_networks[node_id].get_primal_value(dso_models[node_id])

    return primal_value


def check_admm_convergence(operational_planning, consensus_vars, params, consider_shared_ess, debug_flag=False):
    if check_consensus_convergence(operational_planning, consensus_vars, params, consider_shared_ess, debug_flag=debug_flag):
        if check_stationary_convergence(operational_planning, consensus_vars, params, consider_shared_ess):
            print(f'[INFO]\t\t - Converged!')
            return True
    return False


def check_consensus_convergence(operational_planning, consensus_vars, params, consider_shared_ess, debug_flag=False):

    sum_rel_abs_error_vmag, sum_rel_abs_error_pf, sum_rel_abs_error_ess = 0.00, 0.00, 0.00
    num_elems_vmag, num_elems_pf, num_elems_ess = 0, 0, 0
    for node_id in operational_planning.active_distribution_network_nodes:

        s_base = operational_planning.transmission_network.baseMVA

        interface_v_base = operational_planning.transmission_network.get_node_base_kv(node_id)
        interface_transf_rating = operational_planning.distribution_networks[node_id].get_interface_branch_rating()
        if consider_shared_ess:
            shared_ess_idx = operational_planning.transmission_network.get_shared_energy_storage_idx(node_id)
            shared_ess_rating = abs(operational_planning.transmission_network.network.shared_energy_storages[shared_ess_idx].s) * s_base
            if isclose(shared_ess_rating, 0.00, abs_tol=SMALL_TOLERANCE):
                shared_ess_rating = 1.00

        sum_rel_abs_error_vmag += abs(sqrt(consensus_vars['v_sqr']['tso']['current'][node_id]) - sqrt(consensus_vars['v_sqr']['dso']['current'][node_id])) / interface_v_base
        num_elems_vmag += 2

        sum_rel_abs_error_pf += abs(consensus_vars['pf']['tso']['current'][node_id]['p'] - consensus_vars['pf']['dso']['current'][node_id]['p']) / interface_transf_rating
        sum_rel_abs_error_pf += abs(consensus_vars['pf']['tso']['current'][node_id]['q'] - consensus_vars['pf']['dso']['current'][node_id]['q']) / interface_transf_rating
        num_elems_pf += 4

        if consider_shared_ess:
            sum_rel_abs_error_ess += abs(consensus_vars['ess']['tso']['current'][node_id]['p'] - consensus_vars['ess']['dso']['current'][node_id]['p']) / shared_ess_rating
            sum_rel_abs_error_ess += abs(consensus_vars['ess']['tso']['current'][node_id]['q'] - consensus_vars['ess']['dso']['current'][node_id]['q']) / shared_ess_rating
            num_elems_ess += 4

    convergence = True
    if error_within_limits(sum_rel_abs_error_vmag, num_elems_vmag, params.tol['consensus']['v']):
        if error_within_limits(sum_rel_abs_error_pf, num_elems_pf, params.tol['consensus']['pf']):
            if consider_shared_ess:
                if error_within_limits(sum_rel_abs_error_ess, num_elems_ess, params.tol['consensus']['ess']):
                    print('[INFO]\t\t - Consensus constraints ok!')
                else:
                    convergence = False
                    print('[INFO]\t\t - Convergence shared ESS consensus constraints failed. {:.3f} > {:.3f}'.format(sum_rel_abs_error_ess, params.tol['consensus']['ess'] * num_elems_ess))
                    if debug_flag:
                        print_debug_info(operational_planning, consensus_vars, print_ess=True)
        else:
            convergence = False
            print('[INFO]\t\t - Convergence interface PF consensus constraints failed. {:.3f} > {:.3f}'.format(sum_rel_abs_error_pf, params.tol['consensus']['pf'] * num_elems_pf))
            if debug_flag:
                print_debug_info(operational_planning, consensus_vars, print_pf=True)
    else:
        convergence = False
        print('[INFO]\t\t - Convergence interface Vmag consensus constraints failed. {:.3f} > {:.3f}'.format(sum_rel_abs_error_vmag, params.tol['consensus']['v'] * num_elems_vmag))
        if debug_flag:
            print_debug_info(operational_planning, consensus_vars, print_vmag=True)

    return convergence


def check_stationary_convergence(operational_planning, consensus_vars, params, consider_shared_ess):

    rho_tso_v = params.rho['v'][operational_planning.transmission_network.name]
    rho_tso_pf = params.rho['pf'][operational_planning.transmission_network.name]
    rho_tso_ess = params.rho['ess'][operational_planning.transmission_network.name]

    sum_rel_abs_error_vmag, sum_rel_abs_error_pf, sum_rel_abs_error_ess = 0.00, 0.00, 0.00
    num_elems_vmag, num_elems_pf, num_elems_ess = 0, 0, 0
    for node_id in operational_planning.distribution_networks:

        rho_dso_v = params.rho['v'][operational_planning.distribution_networks[node_id].name]
        rho_dso_pf = params.rho['pf'][operational_planning.distribution_networks[node_id].name]
        rho_dso_ess = params.rho['ess'][operational_planning.distribution_networks[node_id].name]

        s_base = operational_planning.transmission_network.baseMVA
        interface_v_base = operational_planning.transmission_network.get_node_base_kv(node_id)
        interface_transf_rating = operational_planning.distribution_networks[node_id].get_interface_branch_rating()
        if consider_shared_ess:
            shared_ess_idx = operational_planning.transmission_network.get_shared_energy_storage_idx(node_id)
            shared_ess_rating = abs(operational_planning.transmission_network.shared_energy_storages[shared_ess_idx].s) * s_base
            if isclose(shared_ess_rating, 0.00, abs_tol=SMALL_TOLERANCE):
                shared_ess_rating = 1.00

        sum_rel_abs_error_vmag += rho_tso_v * abs(sqrt(consensus_vars['v_sqr']['tso']['current'][node_id]) - sqrt(consensus_vars['v_sqr']['tso']['prev'][node_id])) / interface_v_base
        sum_rel_abs_error_vmag += rho_dso_v * abs(sqrt(consensus_vars['v_sqr']['dso']['current'][node_id]) - sqrt(consensus_vars['v_sqr']['dso']['prev'][node_id])) / interface_v_base
        num_elems_vmag += 2

        sum_rel_abs_error_pf += rho_tso_pf * abs(consensus_vars['pf']['tso']['current'][node_id]['p'] - consensus_vars['pf']['tso']['prev'][node_id]['p']) / interface_transf_rating
        sum_rel_abs_error_pf += rho_tso_pf * abs(consensus_vars['pf']['tso']['current'][node_id]['q'] - consensus_vars['pf']['tso']['prev'][node_id]['q']) / interface_transf_rating
        sum_rel_abs_error_pf += rho_dso_pf * abs(consensus_vars['pf']['dso']['current'][node_id]['p'] - consensus_vars['pf']['dso']['prev'][node_id]['p']) / interface_transf_rating
        sum_rel_abs_error_pf += rho_dso_pf * abs(consensus_vars['pf']['dso']['current'][node_id]['q'] - consensus_vars['pf']['dso']['prev'][node_id]['q']) / interface_transf_rating
        num_elems_pf += 4

        if consider_shared_ess:
            sum_rel_abs_error_ess += rho_tso_ess * abs(consensus_vars['ess']['tso']['current'][node_id]['p'] - consensus_vars['ess']['tso']['prev'][node_id]['p']) / shared_ess_rating
            sum_rel_abs_error_ess += rho_tso_ess * abs(consensus_vars['ess']['tso']['current'][node_id]['q'] - consensus_vars['ess']['tso']['prev'][node_id]['q']) / shared_ess_rating
            sum_rel_abs_error_ess += rho_dso_ess * abs(consensus_vars['ess']['dso']['current'][node_id]['p'] - consensus_vars['ess']['dso']['prev'][node_id]['p']) / shared_ess_rating
            sum_rel_abs_error_ess += rho_dso_ess * abs(consensus_vars['ess']['dso']['current'][node_id]['q'] - consensus_vars['ess']['dso']['prev'][node_id]['q']) / shared_ess_rating
            num_elems_ess += 4

    convergence = True
    if error_within_limits(sum_rel_abs_error_vmag, num_elems_vmag, params.tol['stationarity']['v']):
        if error_within_limits(sum_rel_abs_error_pf, num_elems_pf, params.tol['stationarity']['pf']):
            if consider_shared_ess:
                if error_within_limits(sum_rel_abs_error_ess, num_elems_ess, params.tol['stationarity']['ess']):
                    print('[INFO]\t\t - Stationary constraints ok!')
                else:
                    convergence = False
                    print('[INFO]\t\t - Convergence shared ESS stationary constraints failed. {:.3f} > {:.3f}'.format(sum_rel_abs_error_ess, params.tol['stationarity']['ess'] * num_elems_ess))
        else:
            convergence = False
            print('[INFO]\t\t - Convergence interface PF stationary constraints failed. {:.3f} > {:.3f}'.format(sum_rel_abs_error_pf, params.tol['stationarity']['pf'] * num_elems_pf))
    else:
        convergence = False
        print('[INFO]\t\t - Convergence interface Vmag stationary constraints failed. {:.3f} > {:.3f}'.format(sum_rel_abs_error_vmag, params.tol['stationarity']['v'] * num_elems_vmag))

    return convergence


def error_within_limits(sum_abs_error, num_elems, tol):
    if sum_abs_error > tol * num_elems:
        if not isclose(sum_abs_error, tol * num_elems, rel_tol=ADMM_CONVERGENCE_REL_TOL, abs_tol=ADMM_CONVERGENCE_ABS_TOL):
            return False
    return True


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


def print_debug_info(operational_planning, consensus_vars, print_vmag=False, print_pf=False, print_ess=False):
    for node_id in operational_planning.active_distribution_network_nodes:
        if print_vmag:
            print(f"\tNode {node_id}, PF, TSO,  V  {[sqrt(vmag) for vmag in consensus_vars['v_sqr']['tso']['current'][node_id]]}")
            print(f"\tNode {node_id}, PF, DSO,  V  {[sqrt(vmag) for vmag in consensus_vars['v_sqr']['dso']['current'][node_id]]}")
        if print_pf:
            print(f"\tNode {node_id}, PF, TSO,  P {consensus_vars['pf']['tso']['current'][node_id]['p']}")
            print(f"\tNode {node_id}, PF, DSO,  P {consensus_vars['pf']['dso']['current'][node_id]['p']}")
            print(f"\tNode {node_id}, PF, TSO,  Q {consensus_vars['pf']['tso']['current'][node_id]['q']}")
            print(f"\tNode {node_id}, PF, DSO,  Q {consensus_vars['pf']['dso']['current'][node_id]['q']}")
        if print_ess:
            print(f"\tNode {node_id}, ESS, TSO,  P {consensus_vars['ess']['tso']['current'][node_id]['p']}")
            print(f"\tNode {node_id}, ESS, DSO,  P {consensus_vars['ess']['dso']['current'][node_id]['p']}")
            print(f"\tNode {node_id}, ESS, TSO,  Q {consensus_vars['ess']['tso']['current'][node_id]['q']}")
            print(f"\tNode {node_id}, ESS, DSO,  Q {consensus_vars['ess']['dso']['current'][node_id]['q']}")
