import os
import pandas as pd
import pyomo.opt as po
import pyomo.environ as pe
from math import acos, sqrt, tan, atan2
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

    def build_model(self):
        _pre_process_network(self)
        return _build_model(self)

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

    def get_reference_node_id(self):
        for node in self.nodes:
            if node.type == BUS_REF:
                return node.bus_i
        print(f'[ERROR] Network {self.name}. No REF NODE found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_type(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.type
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_base_kv(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.base_kv
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

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

    def get_gen_idx(self, node_id):
        for g in range(len(self.generators)):
            gen = self.generators[g]
            if gen.bus == node_id:
                return g
        print(f'[ERROR] Network {self.name}. No Generator in bus {node_id} found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def compute_series_admittance(self):
        for branch in self.branches:
            branch.g = branch.r / (branch.r ** 2 + branch.x ** 2)
            branch.b = -branch.x / (branch.r ** 2 + branch.x ** 2)

    def perform_network_check(self):
        _perform_network_check(self)


# ======================================================================================================================
#   NETWORK optimization functions
# ======================================================================================================================
def _build_model(network, n=0):

    network.compute_series_admittance()

    model = pe.ConcreteModel()
    model.name = network.name
    params = network.params

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.scenarios_operation = range(len(network.prob_operation_scenarios))
    model.nodes = range(len(network.nodes))
    model.loads = range(len(network.loads))
    model.generators = range(len(network.generators))
    model.branches = range(len(network.branches))
    model.energy_storages = range(len(network.energy_storages))
    model.shared_energy_storages = range(len(network.shared_energy_storages))

    # ------------------------------------------------------------------------------------------------------------------
    # Decision variables
    # - Voltage
    model.e = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=1.0)
    model.f = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    model.e_actual = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=1.0)
    model.f_actual = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    if params.slacks.grid_operation.voltage:
        model.slack_e = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=0.00)
        model.slack_f = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=0.00)
    for i in model.nodes:
        node = network.nodes[i]
        e_lb, e_ub = -node.v_max, node.v_max
        f_lb, f_ub = -node.v_max, node.v_max
        for s_o in model.scenarios_operation:
            if params.slacks.grid_operation.voltage:
                model.slack_e[i, s_o].setub(VMAG_VIOLATION_ALLOWED)
                model.slack_e[i, s_o].setlb(-VMAG_VIOLATION_ALLOWED)
                model.slack_f[i, s_o].setub(VMAG_VIOLATION_ALLOWED)
                model.slack_f[i, s_o].setlb(-VMAG_VIOLATION_ALLOWED)
            if node.type == BUS_REF:
                if network.is_transmission:
                    model.e[i, s_o].setub(e_ub)
                    model.e[i, s_o].setlb(e_lb)
                else:
                    ref_gen_idx = network.get_gen_idx(node.bus_i)
                    vg = network.generators[ref_gen_idx].vg
                    model.e[i, s_o].setub(vg + EQUALITY_TOLERANCE)
                    model.e[i, s_o].setlb(vg - EQUALITY_TOLERANCE)
                    if params.slacks.grid_operation.voltage:
                        model.slack_e[i, s_o].setub(EQUALITY_TOLERANCE)
                        model.slack_e[i, s_o].setlb(-EQUALITY_TOLERANCE)
                model.f[i, s_o].setub(EQUALITY_TOLERANCE)
                model.f[i, s_o].setlb(-EQUALITY_TOLERANCE)
                if params.slacks.grid_operation.voltage:
                    model.slack_f[i, s_o].setub(EQUALITY_TOLERANCE)
                    model.slack_f[i, s_o].setlb(-EQUALITY_TOLERANCE)
            else:
                model.e[i, s_o].setub(e_ub)
                model.e[i, s_o].setlb(e_lb)
                model.f[i, s_o].setub(f_ub)
                model.f[i, s_o].setlb(f_lb)
    if params.slacks.node_balance:
        model.slack_node_balance_p = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=0.00)
        model.slack_node_balance_q = pe.Var(model.nodes, model.scenarios_operation, domain=pe.Reals, initialize=0.00)

    # - Generation
    model.pg = pe.Var(model.generators, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    model.qg = pe.Var(model.generators, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    for g in model.generators:
        generator = network.generators[g]
        pg_ub, pg_lb = generator.pmax, generator.pmin
        qg_ub, qg_lb = generator.qmax, generator.qmin
        for s_o in model.scenarios_operation:
            if generator.status[n]:
                model.pg[g, s_o] = max(pg_lb, 0.00)
                model.qg[g, s_o] = max(qg_lb, 0.00)
                model.pg[g, s_o].setub(pg_ub)
                model.pg[g, s_o].setlb(pg_lb)
                model.qg[g, s_o].setub(qg_ub)
                model.qg[g, s_o].setlb(qg_lb)
            else:
                model.pg[g, s_o].setub(EQUALITY_TOLERANCE)
                model.pg[g, s_o].setlb(-EQUALITY_TOLERANCE)
                model.qg[g, s_o].setub(EQUALITY_TOLERANCE)
                model.qg[g, s_o].setlb(-EQUALITY_TOLERANCE)
    if params.rg_curt:
        model.sg_abs = pe.Var(model.generators, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.sg_sqr = pe.Var(model.generators, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.sg_curt = pe.Var(model.generators, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        for g in model.generators:
            generator = network.generators[g]
            for s_o in model.scenarios_operation:
                if generator.is_curtaillable():
                    # - Renewable Generation
                    init_sg = 0.0
                    if generator.status[n]:
                        init_sg = sqrt(generator.pg[s_o][n] ** 2 + generator.qg[s_o][n] ** 2)
                    model.sg_abs[g, s_o].setub(init_sg)
                    model.sg_sqr[g, s_o].setub(init_sg ** 2)
                    model.sg_curt[g, s_o].setub(init_sg)
                else:
                    model.sg_abs[g, s_o].setub(EQUALITY_TOLERANCE)
                    model.sg_sqr[g, s_o].setub(EQUALITY_TOLERANCE)
                    model.sg_curt[g, s_o].setub(EQUALITY_TOLERANCE)

    # - Branch power flows (squared) -- used in branch limits
    model.flow_ij_sqr = pe.Var(model.branches, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slacks.grid_operation.branch_flow:
        model.slack_flow_ij_sqr = pe.Var(model.branches, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
    for b in model.branches:
        for s_o in model.scenarios_operation:
            if network.branches[b].status:
                if params.slacks.grid_operation.branch_flow:
                    rating = network.branches[b].rate / network.baseMVA
                    model.slack_flow_ij_sqr[b, s_o].setub(SIJ_VIOLATION_ALLOWED * rating)
            else:
                model.flow_ij_sqr[b, s_o].setub(EQUALITY_TOLERANCE)
                if params.slacks.grid_operation.branch_flow:
                    model.slack_flow_ij_sqr[b, s_o].setub(EQUALITY_TOLERANCE)

    # - Loads
    model.pc = pe.Var(model.loads, model.scenarios_operation, domain=pe.Reals)
    model.qc = pe.Var(model.loads, model.scenarios_operation, domain=pe.Reals)
    for c in model.loads:
        load = network.loads[c]
        for s_o in model.scenarios_operation:
            model.pc[c, s_o].setub(load.pd[s_o][n] + EQUALITY_TOLERANCE)
            model.pc[c, s_o].setlb(load.pd[s_o][n] - EQUALITY_TOLERANCE)
            model.qc[c, s_o].setub(load.qd[s_o][n] + EQUALITY_TOLERANCE)
            model.qc[c, s_o].setlb(load.qd[s_o][n] - EQUALITY_TOLERANCE)
    if params.fl_reg:
        model.flex_p_up = pe.Var(model.loads, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.flex_p_down = pe.Var(model.loads, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        for c in model.loads:
            load = network.loads[c]
            for s_o in model.scenarios_operation:
                if load.fl_reg:
                    flex_up = load.flexibility.upward[n]
                    flex_down = load.flexibility.downward[n]
                    model.flex_p_up[c, s_o].setub(abs(flex_up))
                    model.flex_p_down[c, s_o].setub(abs(flex_down))
                else:
                    model.flex_p_up[c, s_o].setub(EQUALITY_TOLERANCE)
                    model.flex_p_down[c, s_o].setub(EQUALITY_TOLERANCE)
    if params.l_curt:
        model.pc_curt_down = pe.Var(model.loads, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.pc_curt_up = pe.Var(model.loads, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.qc_curt_down = pe.Var(model.loads, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.qc_curt_up = pe.Var(model.loads, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        for c in model.loads:
            load = network.loads[c]
            for s_o in model.scenarios_operation:
                if load.pd[s_o][n] >= 0.00:
                    model.pc_curt_down[c, s_o].setub(abs(load.pd[s_o][n]))
                    model.pc_curt_up[c, s_o].setub(EQUALITY_TOLERANCE)
                else:
                    model.pc_curt_up[c, s_o].setub(abs(load.pd[s_o][n]))
                    model.pc_curt_down[c, s_o].setub(EQUALITY_TOLERANCE)

                if load.qd[s_o][n] >= 0.00:
                    model.qc_curt_down[c, s_o].setub(abs(load.qd[s_o][n]))
                    model.qc_curt_up[c, s_o].setub(EQUALITY_TOLERANCE)
                else:
                    model.qc_curt_up[c, s_o].setub(abs(load.qd[s_o][n]))
                    model.qc_curt_down[c, s_o].setub(EQUALITY_TOLERANCE)

    # - Transformers
    model.r = pe.Var(model.branches, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=1.0)
    for i in model.branches:
        branch = network.branches[i]
        for s_o in model.scenarios_operation:
            if branch.is_transformer:
                # - Transformer
                if params.transf_reg and branch.vmag_reg:
                    model.r[i, s_o].setub(TRANSFORMER_MAXIMUM_RATIO)
                    model.r[i, s_o].setlb(TRANSFORMER_MINIMUM_RATIO)
                else:
                    model.r[i, s_o].setub(branch.ratio + EQUALITY_TOLERANCE)
                    model.r[i, s_o].setlb(branch.ratio - EQUALITY_TOLERANCE)
            else:
                model.r[i, s_o].setub(1.00 + EQUALITY_TOLERANCE)
                model.r[i, s_o].setlb(1.00 - EQUALITY_TOLERANCE)

    # - Energy Storage devices
    if params.es_reg:
        model.es_soc = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.00)
        model.es_sch = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pch = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_qch = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
        model.es_sdch = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pdch = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_qdch = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
        for e in model.energy_storages:
            energy_storage = network.energy_storages[e]
            for s_o in model.scenarios_operation:
                model.es_soc[e, s_o] = energy_storage.e_init
                model.es_soc[e, s_o].setlb(energy_storage.e_min)
                model.es_soc[e, s_o].setub(energy_storage.e_max)
                model.es_sch[e, s_o].setub(energy_storage.s)
                model.es_pch[e, s_o].setub(energy_storage.s)
                model.es_qch[e, s_o].setub(energy_storage.s)
                model.es_qch[e, s_o].setlb(-energy_storage.s)
                model.es_sdch[e, s_o].setub(energy_storage.s)
                model.es_pdch[e, s_o].setub(energy_storage.s)
                model.es_qdch[e, s_o].setub(energy_storage.s)
                model.es_qdch[e, s_o].setlb(-energy_storage.s)

        if params.slacks.ess.complementarity:
            model.slack_es_comp = pe.Var(model.energy_storages, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        if params.slacks.ess.charging:
            model.slack_es_ch = pe.Var(model.shared_energy_storages, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
            model.slack_es_dch = pe.Var(model.shared_energy_storages, model.scenarios_operation, domain=pe.Reals, initialize=0.0)

    # - Shared Energy Storage devices
    model.shared_es_s_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)
    model.shared_es_e_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)
    model.shared_es_s_rated_fixed = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)          # Benders' -- used to get the dual variables (sensitivities)
    model.shared_es_e_rated_fixed = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)          # (...)
    model.shared_es_soc = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.shared_es_sch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_pch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_qch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.shared_es_sdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_pdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_qdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.shared_es_pnet = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.shared_es_qnet = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    for e in model.shared_energy_storages:
        shared_energy_storage = network.shared_energy_storages[e]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.shared_es_soc[e, s_m, s_o, p] = shared_energy_storage.e * ENERGY_STORAGE_RELATIVE_INIT_SOC
    if params.slacks.shared_ess.complementarity:
        model.slack_shared_es_comp = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slacks.shared_ess.charging:
        model.slack_shared_es_ch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
        model.slack_shared_es_dch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Voltage
    model.voltage_cons = pe.ConstraintList()
    for i in model.nodes:
        node = network.nodes[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    # e_actual and f_actual definition
                    e_actual = model.e[i, s_m, s_o, p]
                    f_actual = model.f[i, s_m, s_o, p]
                    if params.slacks.grid_operation.voltage:
                        e_actual += model.slack_e[i, s_m, s_o, p]
                        f_actual += model.slack_f[i, s_m, s_o, p]

                    model.voltage_cons.add(model.e_actual[i, s_m, s_o, p] == e_actual)
                    model.voltage_cons.add(model.f_actual[i, s_m, s_o, p] == f_actual)

                    # voltage magnitude constraints
                    if node.type == BUS_PV:
                        if params.enforce_vg:
                            # - Enforce voltage controlled bus
                            gen_idx = network.get_gen_idx(node.bus_i)
                            vg = network.generators[gen_idx].vg
                            e = model.e[i, s_m, s_o, p]
                            f = model.f[i, s_m, s_o, p]
                            model.voltage_cons.add(e ** 2 + f ** 2 <= vg[p] ** 2 + EQUALITY_TOLERANCE)
                            model.voltage_cons.add(e ** 2 + f ** 2 >= vg[p] ** 2 - EQUALITY_TOLERANCE)
                        else:
                            # - Voltage at the bus is not controlled
                            e = model.e[i, s_m, s_o, p]
                            f = model.f[i, s_m, s_o, p]
                            model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min**2)
                            model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max**2)
                    else:
                        e = model.e[i, s_m, s_o, p]
                        f = model.f[i, s_m, s_o, p]
                        model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min**2)
                        model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max**2)

    model.generation_apparent_power = pe.ConstraintList()
    model.generation_power_factor = pe.ConstraintList()
    if params.rg_curt:
        for g in model.generators:
            generator = network.generators[g]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        if generator.is_curtaillable():
                            init_sg = 0.0
                            if generator.status[p]:
                                init_sg = sqrt(generator.pg[s_o][p] ** 2 + generator.qg[s_o][p] ** 2)
                            model.generation_apparent_power.add(model.sg_sqr[g, s_m, s_o, p] <= model.pg[g, s_m, s_o, p] ** 2 + model.qg[g, s_m, s_o, p] ** 2 + EQUALITY_TOLERANCE)
                            model.generation_apparent_power.add(model.sg_sqr[g, s_m, s_o, p] >= model.pg[g, s_m, s_o, p] ** 2 + model.qg[g, s_m, s_o, p] ** 2 - EQUALITY_TOLERANCE)
                            model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o, p] ** 2 <= model.sg_sqr[g, s_m, s_o, p] + EQUALITY_TOLERANCE)
                            model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o, p] ** 2 >= model.sg_sqr[g, s_m, s_o, p] - EQUALITY_TOLERANCE)
                            model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o, p] <= init_sg - model.sg_curt[g, s_m, s_o, p] + EQUALITY_TOLERANCE)
                            model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o, p] >= init_sg - model.sg_curt[g, s_m, s_o, p] - EQUALITY_TOLERANCE)
                            if generator.power_factor_control:
                                # Power factor control, variable phi
                                max_phi = acos(generator.max_pf)
                                min_phi = acos(generator.min_pf)
                                model.generation_power_factor.add(model.qg[g, s_m, s_o, p] <= tan(max_phi) * model.pg[g, s_m, s_o, p])
                                model.generation_power_factor.add(model.qg[g, s_m, s_o, p] >= tan(min_phi) * model.pg[g, s_m, s_o, p])
                            else:
                                # No power factor control, maintain given phi
                                phi = atan2(generator.qg[s_o][p], generator.pg[s_o][p])
                                model.generation_power_factor.add(model.qg[g, s_m, s_o, p] <= tan(phi) * model.pg[g, s_m, s_o, p])
                                model.generation_power_factor.add(model.qg[g, s_m, s_o, p] >= tan(phi) * model.pg[g, s_m, s_o, p])

    # - Flexible Loads -- Daily energy balance
    if params.fl_reg:
        model.fl_p_balance = pe.ConstraintList()
        for c in model.loads:
            if network.loads[c].fl_reg:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        p_up, p_down = 0.0, 0.0
                        for p in model.periods:
                            p_up += model.flex_p_up[c, s_m, s_o, p]
                            p_down += model.flex_p_down[c, s_m, s_o, p]
                        if params.slacks.flexibility.day_balance:
                            model.fl_p_balance.add(p_up == p_down + model.slack_flex_p_balance[c, s_m, s_o])
                        else:
                            model.fl_p_balance.add(p_up <= p_down + EQUALITY_TOLERANCE)
                            model.fl_p_balance.add(p_up >= p_down - EQUALITY_TOLERANCE)

    # - Energy Storage constraints
    if params.es_reg:

        model.energy_storage_balance = pe.ConstraintList()
        model.energy_storage_operation = pe.ConstraintList()
        model.energy_storage_day_balance = pe.ConstraintList()
        model.energy_storage_ch_dch_exclusion = pe.ConstraintList()

        for e in model.energy_storages:

            energy_storage = network.energy_storages[e]
            soc_init = energy_storage.e_init
            soc_final = energy_storage.e_init
            eff_charge = energy_storage.eff_ch
            eff_discharge = energy_storage.eff_dch
            max_phi = acos(energy_storage.max_pf)
            min_phi = acos(energy_storage.min_pf)

            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:

                        sch = model.es_sch[e, s_m, s_o, p]
                        pch = model.es_pch[e, s_m, s_o, p]
                        qch = model.es_qch[e, s_m, s_o, p]
                        sdch = model.es_sdch[e, s_m, s_o, p]
                        pdch = model.es_pdch[e, s_m, s_o, p]
                        qdch = model.es_qdch[e, s_m, s_o, p]

                        # ESS operation
                        model.energy_storage_operation.add(qch <= tan(max_phi) * pch)
                        model.energy_storage_operation.add(qch >= tan(min_phi) * pch)
                        model.energy_storage_operation.add(qdch <= tan(max_phi) * pdch)
                        model.energy_storage_operation.add(qdch >= tan(min_phi) * pdch)

                        if params.slacks.ess.charging:
                            model.energy_storage_operation.add(sch ** 2 == pch ** 2 + qch ** 2 + model.slack_es_ch[e, s_m, s_o, p])
                            model.energy_storage_operation.add(sdch ** 2 == pdch ** 2 + qdch ** 2 + model.slack_es_dch[e, s_m, s_o, p])
                        else:
                            model.energy_storage_operation.add(sch ** 2 <= pch ** 2 + qch ** 2 + EQUALITY_TOLERANCE)
                            model.energy_storage_operation.add(sch ** 2 >= pch ** 2 + qch ** 2 - EQUALITY_TOLERANCE)
                            model.energy_storage_operation.add(sdch ** 2 <= pdch ** 2 + qdch ** 2 + EQUALITY_TOLERANCE)
                            model.energy_storage_operation.add(sdch ** 2 >= pdch ** 2 + qdch ** 2 - EQUALITY_TOLERANCE)

                        # Charging/discharging complementarity constraints
                        if params.slacks.ess.complementarity:
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch == model.slack_es_comp[e, s_m, s_o, p])
                        else:
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch <= EQUALITY_TOLERANCE)

                        # State-of-Charge
                        soc_prev = soc_init
                        if p > 0:
                            soc_prev = model.es_soc[e, s_m, s_o, p - 1]

                        model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] <= soc_prev + (sch * eff_charge - sdch / eff_discharge) + EQUALITY_TOLERANCE)
                        model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] >= soc_prev + (sch * eff_charge - sdch / eff_discharge) - EQUALITY_TOLERANCE)

                    if params.slacks.ess.day_balance:
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] == soc_final + model.slack_es_soc_final[e, s_m, s_o])
                    else:
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] <= soc_final + EQUALITY_TOLERANCE)
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] >= soc_final - EQUALITY_TOLERANCE)

    # - Shared Energy Storage constraints
    model.shared_energy_storage_balance = pe.ConstraintList()
    model.shared_energy_storage_operation = pe.ConstraintList()
    model.shared_energy_storage_day_balance = pe.ConstraintList()
    model.shared_energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.shared_energy_storage_s_sensitivities = pe.ConstraintList()
    model.shared_energy_storage_e_sensitivities = pe.ConstraintList()
    for e in model.shared_energy_storages:

        shared_energy_storage = network.shared_energy_storages[e]
        eff_charge = shared_energy_storage.eff_ch
        eff_discharge = shared_energy_storage.eff_dch
        max_phi = acos(shared_energy_storage.max_pf)
        min_phi = acos(shared_energy_storage.min_pf)

        s_max = model.shared_es_s_rated[e]
        soc_max = model.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
        soc_min = model.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
        soc_init = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
        soc_final = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    sch = model.shared_es_sch[e, s_m, s_o, p]
                    pch = model.shared_es_pch[e, s_m, s_o, p]
                    qch = model.shared_es_qch[e, s_m, s_o, p]
                    sdch = model.shared_es_sdch[e, s_m, s_o, p]
                    pdch = model.shared_es_pdch[e, s_m, s_o, p]
                    qdch = model.shared_es_qdch[e, s_m, s_o, p]

                    # ESS operation
                    model.shared_energy_storage_operation.add(sch <= s_max)
                    model.shared_energy_storage_operation.add(pch <= s_max)
                    model.shared_energy_storage_operation.add(qch <= s_max)
                    model.shared_energy_storage_operation.add(qch <= tan(max_phi) * pch)
                    model.shared_energy_storage_operation.add(qch >= tan(min_phi) * pch)

                    model.shared_energy_storage_operation.add(sdch <= s_max)
                    model.shared_energy_storage_operation.add(pdch <= s_max)
                    model.shared_energy_storage_operation.add(qdch <= s_max)
                    model.shared_energy_storage_operation.add(qdch <= tan(max_phi) * pdch)
                    model.shared_energy_storage_operation.add(qdch >= tan(min_phi) * pdch)

                    # Pnet and Qnet definition
                    model.shared_energy_storage_operation.add(model.shared_es_pnet[e, s_m, s_o, p] <= pch - pdch + EQUALITY_TOLERANCE)
                    model.shared_energy_storage_operation.add(model.shared_es_pnet[e, s_m, s_o, p] >= pch - pdch - EQUALITY_TOLERANCE)
                    model.shared_energy_storage_operation.add(model.shared_es_qnet[e, s_m, s_o, p] <= qch - qdch + EQUALITY_TOLERANCE)
                    model.shared_energy_storage_operation.add(model.shared_es_qnet[e, s_m, s_o, p] >= qch - qdch - EQUALITY_TOLERANCE)

                    model.shared_energy_storage_operation.add(model.shared_es_soc[e, s_m, s_o, p] <= soc_max)
                    model.shared_energy_storage_operation.add(model.shared_es_soc[e, s_m, s_o, p] >= soc_min)

                    if params.slacks.shared_ess.charging:
                        model.shared_energy_storage_operation.add(sch ** 2 == pch ** 2 + qch ** 2 + model.slack_shared_es_ch[e, s_m, s_o, p])
                        model.shared_energy_storage_operation.add(sdch ** 2 == pdch ** 2 + qdch ** 2 + model.slack_shared_es_dch[e, s_m, s_o, p])
                    else:
                        model.shared_energy_storage_operation.add(sch ** 2 <= pch ** 2 + qch ** 2 + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(sch ** 2 >= pch ** 2 + qch ** 2 - EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(sdch ** 2 <= pdch ** 2 + qdch ** 2 + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(sdch ** 2 >= pdch ** 2 + qdch ** 2 - EQUALITY_TOLERANCE)

                    # Charging/discharging complementarity constraints
                    if params.slacks.shared_ess.complementarity:
                        model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch == model.slack_shared_es_comp[e, s_m, s_o, p])
                    else:
                        model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch <= EQUALITY_TOLERANCE)

                    # State-of-Charge
                    soc_prev = soc_init
                    if p > 0:
                        soc_prev = model.shared_es_soc[e, s_m, s_o, p - 1]
                    model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] <= soc_prev + (sch * eff_charge - sdch / eff_discharge) + EQUALITY_TOLERANCE)
                    model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] >= soc_prev + (sch * eff_charge - sdch / eff_discharge) - EQUALITY_TOLERANCE)

                # Day balance
                if params.slacks.shared_ess.day_balance:
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] <= soc_final + model.slack_shared_es_soc_final[e, s_m, s_o])
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] >= soc_final + model.slack_shared_es_soc_final[e, s_m, s_o])
                else:
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] <= soc_final + EQUALITY_TOLERANCE)
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] >= soc_final - EQUALITY_TOLERANCE)

        model.shared_energy_storage_s_sensitivities.add(model.shared_es_s_rated[e] <= model.shared_es_s_rated_fixed[e])
        model.shared_energy_storage_e_sensitivities.add(model.shared_es_e_rated[e] <= model.shared_es_e_rated_fixed[e])

    # - Node Balance constraints
    model.node_balance_cons_p = pe.ConstraintList()
    model.node_balance_cons_q = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for i in range(len(network.nodes)):

                    node = network.nodes[i]

                    Pd = 0.00
                    Qd = 0.00
                    for c in model.loads:
                        if network.loads[c].bus == node.bus_i:
                            Pd += model.pc[c, s_m, s_o, p]
                            Qd += model.qc[c, s_m, s_o, p]
                            if params.fl_reg and network.loads[c].fl_reg:
                                Pd += (model.flex_p_up[c, s_m, s_o, p] - model.flex_p_down[c, s_m, s_o, p])
                            if params.l_curt:
                                Pd -= (model.pc_curt_down[c, s_m, s_o, p] - model.pc_curt_up[c, s_m, s_o, p])
                                Qd -= (model.qc_curt_down[c, s_m, s_o, p] - model.qc_curt_up[c, s_m, s_o, p])
                    if params.es_reg:
                        for e in model.energy_storages:
                            if network.energy_storages[e].bus == node.bus_i:
                                Pd += (model.es_pch[e, s_m, s_o, p] - model.es_pdch[e, s_m, s_o, p])
                                Qd += (model.es_qch[e, s_m, s_o, p] - model.es_qdch[e, s_m, s_o, p])
                    for e in model.shared_energy_storages:
                        if network.shared_energy_storages[e].bus == node.bus_i:
                            Pd += (model.shared_es_pch[e, s_m, s_o, p] - model.shared_es_pdch[e, s_m, s_o, p])
                            Qd += (model.shared_es_qch[e, s_m, s_o, p] - model.shared_es_qdch[e, s_m, s_o, p])

                    Pg = 0.0
                    Qg = 0.0
                    for g in model.generators:
                        generator = network.generators[g]
                        if generator.bus == node.bus_i:
                            Pg += model.pg[g, s_m, s_o, p]
                            Qg += model.qg[g, s_m, s_o, p]

                    ei = model.e_actual[i, s_m, s_o, p]
                    fi = model.f_actual[i, s_m, s_o, p]

                    Pi = node.gs * (ei ** 2 + fi ** 2)
                    Qi = -node.bs * (ei ** 2 + fi ** 2)
                    for b in range(len(network.branches)):
                        branch = network.branches[b]
                        if branch.fbus == node.bus_i or branch.tbus == node.bus_i:

                            rij = model.r[b, s_m, s_o, p]
                            if not branch.is_transformer:
                                rij = 1.00

                            if branch.fbus == node.bus_i:
                                fnode_idx = network.get_node_idx(branch.fbus)
                                tnode_idx = network.get_node_idx(branch.tbus)

                                ei = model.e_actual[fnode_idx, s_m, s_o, p]
                                fi = model.f_actual[fnode_idx, s_m, s_o, p]
                                ej = model.e_actual[tnode_idx, s_m, s_o, p]
                                fj = model.f_actual[tnode_idx, s_m, s_o, p]

                                Pi += branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                                Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                                Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2) * rij ** 2
                                Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))
                            else:
                                fnode_idx = network.get_node_idx(branch.tbus)
                                tnode_idx = network.get_node_idx(branch.fbus)

                                ei = model.e_actual[fnode_idx, s_m, s_o, p]
                                fi = model.f_actual[fnode_idx, s_m, s_o, p]
                                ej = model.e_actual[tnode_idx, s_m, s_o, p]
                                fj = model.f_actual[tnode_idx, s_m, s_o, p]

                                Pi += branch.g * (ei ** 2 + fi ** 2)
                                Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                                Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2)
                                Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

                    if params.slacks.node_balance:
                        model.node_balance_cons_p.add(Pg == Pd + Pi + model.slack_node_balance_p[i, s_m, s_o, p])
                        model.node_balance_cons_q.add(Qg == Qd + Qi + model.slack_node_balance_q[i, s_m, s_o, p])
                    else:
                        model.node_balance_cons_p.add(Pg <= Pd + Pi + EQUALITY_TOLERANCE)
                        model.node_balance_cons_p.add(Pg >= Pd + Pi - EQUALITY_TOLERANCE)
                        model.node_balance_cons_q.add(Qg <= Qd + Qi + EQUALITY_TOLERANCE)
                        model.node_balance_cons_q.add(Qg >= Qd + Qi - EQUALITY_TOLERANCE)

    # - Branch Power Flow constraints (current)
    model.branch_power_flow_cons = pe.ConstraintList()
    model.branch_power_flow_lims = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for b in model.branches:

                    branch = network.branches[b]
                    rating = branch.rate / network.baseMVA
                    if rating == 0.0:
                        rating = BRANCH_UNKNOWN_RATING
                    fnode_idx = network.get_node_idx(branch.fbus)
                    tnode_idx = network.get_node_idx(branch.tbus)

                    rij = model.r[b, s_m, s_o, p]
                    if not branch.is_transformer:
                        rij = 1.00
                    ei = model.e_actual[fnode_idx, s_m, s_o, p]
                    fi = model.f_actual[fnode_idx, s_m, s_o, p]
                    ej = model.e_actual[tnode_idx, s_m, s_o, p]
                    fj = model.f_actual[tnode_idx, s_m, s_o, p]

                    flow_ij_sqr = 0.00

                    if params.branch_limit_type == BRANCH_LIMIT_CURRENT:

                        bij_sh = branch.b_sh * 0.50

                        iij_sqr = (branch.g ** 2 + branch.b ** 2) * (((rij ** 2) * ei - rij * ej) ** 2 + ((rij ** 2) * fi - rij * fj) ** 2)
                        iij_sqr += bij_sh ** 2 * (ei ** 2 + fi ** 2)
                        iij_sqr += 2 * branch.g * bij_sh * (((rij ** 2) * fi - rij * fj) * ei - ((rij ** 2) * ei - rij * ej) * fi)
                        iij_sqr += 2 * branch.b * bij_sh * (((rij ** 2) * ei - rij * ej) * ei + ((rij ** 2) * fi - rij * fj) * fi)
                        flow_ij_sqr = iij_sqr

                        # Previous (approximation?)
                        # iji_sqr = (branch.g ** 2 + branch.b ** 2) * ((ej - rij * ei) ** 2 + (fj - rij * fi) ** 2)
                        # iji_sqr += bij_sh ** 2 * (ej ** 2 + fj ** 2)
                        # iji_sqr += 2 * branch.g * bij_sh * ((fj - rij * fi) * ej - (ej - rij * ei) * fj)
                        # iji_sqr += 2 * branch.b * bij_sh * ((ej - rij * ei) * ej + (fj - rij * fi) * fj)

                    elif params.branch_limit_type == BRANCH_LIMIT_APPARENT_POWER:

                        pij = branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                        pij -= branch.g * (ei * ej + fi * fj) * rij
                        pij -= branch.b * (fi * ej - ei * fj) * rij
                        qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2) * rij ** 2
                        qij += branch.b * (ei * ej + fi * fj) * rij
                        qij -= branch.g * (fi * ej - ei * fj) * rij
                        sij_sqr = pij ** 2 + qij ** 2
                        flow_ij_sqr = sij_sqr

                        # Without rij
                        # pji = branch.g * (ej ** 2 + fj ** 2)
                        # pji -= branch.g * (ej * ei + fj * fi) * rij
                        # pji -= branch.b * (fj * ei - ej * fi) * rij
                        # qji = - (branch.b + branch.b_sh * 0.50) * (ej ** 2 + fj ** 2)
                        # qji += branch.b * (ej * ei + fj * fi) * rij
                        # qji -= branch.g * (fj * ei - ej * fi) * rij
                        # sji_sqr = pji ** 2 + qji ** 2

                    elif params.branch_limit_type == BRANCH_LIMIT_MIXED:

                        if branch.is_transformer:
                            pij = branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                            pij -= branch.g * (ei * ej + fi * fj) * rij
                            pij -= branch.b * (fi * ej - ei * fj) * rij
                            qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2) * rij ** 2
                            qij += branch.b * (ei * ej + fi * fj) * rij
                            qij -= branch.g * (fi * ej - ei * fj) * rij
                            sij_sqr = pij ** 2 + qij ** 2
                            flow_ij_sqr = sij_sqr
                        else:
                            bij_sh = branch.b_sh * 0.50
                            iij_sqr = (branch.g ** 2 + branch.b ** 2) * (((rij ** 2) * ei - rij * ej) ** 2 + ((rij ** 2) * fi - rij * fj) ** 2)
                            iij_sqr += bij_sh ** 2 * (ei ** 2 + fi ** 2)
                            iij_sqr += 2 * branch.g * bij_sh * (((rij ** 2) * fi - rij * fj) * ei - ((rij ** 2) * ei - rij * ej) * fi)
                            iij_sqr += 2 * branch.b * bij_sh * (((rij ** 2) * ei - rij * ej) * ei + ((rij ** 2) * fi - rij * fj) * fi)
                            flow_ij_sqr = iij_sqr

                    # Flow_ij, definition
                    model.branch_power_flow_cons.add(model.flow_ij_sqr[b, s_m, s_o, p] <= flow_ij_sqr + EQUALITY_TOLERANCE)
                    model.branch_power_flow_cons.add(model.flow_ij_sqr[b, s_m, s_o, p] >= flow_ij_sqr - EQUALITY_TOLERANCE)

                    # Branch flow limits
                    if branch.status:
                        if params.slacks.grid_operation.branch_flow:
                            model.branch_power_flow_lims.add(model.flow_ij_sqr[b, s_m, s_o, p] <= rating ** 2 + model.slack_flow_ij_sqr[b, s_m, s_o, p])
                        else:
                            model.branch_power_flow_lims.add(model.flow_ij_sqr[b, s_m, s_o, p] <= rating ** 2)

    # ------------------------------------------------------------------------------------------------------------------
    # Costs (penalties)
    # Note: defined as variables (bus fixed) so that they can be changed later, if needed
    model.penalty_ess_usage = pe.Var(domain=pe.NonNegativeReals)
    model.penalty_ess_usage.fix(PENALTY_ESS_USAGE)
    if params.obj_type == OBJ_MIN_COST:
        model.cost_res_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.cost_load_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.cost_res_curtailment.fix(COST_GENERATION_CURTAILMENT)
        model.cost_load_curtailment.fix(COST_CONSUMPTION_CURTAILMENT)
    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        model.penalty_gen_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.penalty_load_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.penalty_flex_usage = pe.Var(domain=pe.NonNegativeReals)
        model.penalty_gen_curtailment.fix(PENALTY_GENERATION_CURTAILMENT)
        model.penalty_load_curtailment.fix(PENALTY_LOAD_CURTAILMENT)
        model.penalty_flex_usage.fix(PENALTY_FLEXIBILITY_USAGE)
    else:
        print(f'[ERROR] Unrecognized or invalid objective. Objective = {params.obj_type}. Exiting...')
        exit(ERROR_NETWORK_MODEL)

    # Objective Function
    obj = 0.0
    if params.obj_type == OBJ_MIN_COST:

        # Cost minimization
        c_p = network.cost_energy_p
        c_flex = network.cost_flex
        for s_m in model.scenarios_market:
            omega_market = network.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0
                omega_oper = network.prob_operation_scenarios[s_o]

                # Generation
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        if (not network.is_transmission) and network.generators[g].gen_type == GEN_REFERENCE:
                            continue
                        for p in model.periods:
                            pg = model.pg[g, s_m, s_o, p]
                            obj_scenario += c_p[s_m][p] * network.baseMVA * pg

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        for p in model.periods:
                            flex_p_up = model.flex_p_up[c, s_m, s_o, p]
                            flex_p_down = model.flex_p_down[c, s_m, s_o, p]
                            obj_scenario += c_flex[s_m][p] * network.baseMVA * (flex_p_down + flex_p_up)

                # Load curtailment
                if params.l_curt:
                    for c in model.loads:
                        for p in model.periods:
                            pc_curt = (model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p])
                            qc_curt = (model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p])
                            obj_scenario += model.cost_load_curtailment * network.baseMVA * (pc_curt + qc_curt)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        if network.generators[g].is_curtaillable():
                            for p in model.periods:
                                sg_curt = model.sg_curt[g, s_m, s_o, p]
                                obj_scenario += model.cost_res_curtailment * network.baseMVA * sg_curt

                # ESS utilization
                if params.es_reg:
                    for e in model.energy_storages:
                        for p in model.periods:
                            sch = model.es_sch[e, s_m, s_o, p]
                            sdch = model.es_sdch[e, s_m, s_o, p]
                            obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                # Shared ESS utilization
                for e in model.shared_energy_storages:
                    for p in model.periods:
                        sch = model.shared_es_sch[e, s_m, s_o, p]
                        sdch = model.shared_es_sdch[e, s_m, s_o, p]
                        obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                obj += obj_scenario * omega_market * omega_oper
    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        # Congestion Management
        for s_m in model.scenarios_market:

            omega_market = network.prob_market_scenarios[s_m]

            for s_o in model.scenarios_operation:

                omega_oper = network.prob_operation_scenarios[s_o]

                obj_scenario = 0.0

                # Generation curtailment
                # if params.rg_curt:
                #     for g in model.generators:
                #         for p in model.periods:
                #             sg_curt = model.sg_curt[g, s_m, s_o, p]
                #             obj_scenario += model.penalty_gen_curtailment * network.baseMVA * sg_curt

                # Load curtailment
                if params.l_curt:
                    for c in model.loads:
                        for p in model.periods:
                            pc_curt = (model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p])
                            qc_curt = (model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p])
                            obj_scenario += model.penalty_load_curtailment * network.baseMVA * (pc_curt + qc_curt)

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        for p in model.periods:
                            flex_p_up = model.flex_p_up[c, s_m, s_o, p]
                            flex_p_down = model.flex_p_down[c, s_m, s_o, p]
                            obj_scenario += model.penalty_flex_usage * network.baseMVA * (flex_p_down + flex_p_up)

                # ESS utilization
                if params.es_reg:
                    for e in model.energy_storages:
                        for p in model.periods:
                            sch = model.es_sch[e, s_m, s_o, p]
                            sdch = model.es_sdch[e, s_m, s_o, p]
                            obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                # Shared ESS utilization
                for e in model.shared_energy_storages:
                    for p in model.periods:
                        sch = model.shared_es_sch[e, s_m, s_o, p]
                        sdch = model.shared_es_sdch[e, s_m, s_o, p]
                        obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                obj += obj_scenario * omega_market * omega_oper

    # Slacks grid operation
    for s_m in model.scenarios_market:

        omega_market = network.prob_market_scenarios[s_m]

        for s_o in model.scenarios_operation:

            omega_oper = network.prob_operation_scenarios[s_o]

            # Voltage slacks
            if params.slacks.grid_operation.voltage:
                for i in model.nodes:
                    for p in model.periods:
                        slack_e_sqr = model.slack_e[i, s_m, s_o, p] ** 2
                        slack_f_sqr = model.slack_f[i, s_m, s_o, p] ** 2
                        obj += PENALTY_VOLTAGE * network.baseMVA * omega_market * omega_oper * (slack_e_sqr + slack_f_sqr)

            # Branch power flow slacks
            if params.slacks.grid_operation.branch_flow:
                for b in model.branches:
                    for p in model.periods:
                        slack_flow_ij_sqr = (model.slack_flow_ij_sqr[b, s_m, s_o, p])
                        obj += PENALTY_CURRENT * network.baseMVA * omega_market * omega_oper * slack_flow_ij_sqr

    # Operation slacks
    for s_m in model.scenarios_market:

        omega_market = network.prob_market_scenarios[s_m]

        for s_o in model.scenarios_operation:

            omega_oper = network.prob_operation_scenarios[s_o]

            # Node balance
            if params.slacks.node_balance:
                for i in model.nodes:
                    for p in model.periods:
                        slack_p_sqr = model.slack_node_balance_p[i, s_m, s_o, p] ** 2
                        slack_q_sqr = model.slack_node_balance_q[i, s_m, s_o, p] ** 2
                        obj += PENALTY_NODE_BALANCE * network.baseMVA * omega_market * omega_oper * (slack_p_sqr + slack_q_sqr)

            # ESS slacks
            if params.es_reg:
                for e in model.energy_storages:
                    for p in model.periods:
                        if params.slacks.ess.complementarity:
                            slack_comp = model.slack_es_comp[e, s_m, s_o, p]
                            obj += PENALTY_ESS * network.baseMVA * omega_market * omega_oper * slack_comp
                        if params.slacks.ess.charging:
                            slack_ch_sqr = model.slack_es_ch[e, s_m, s_o, p] ** 2
                            slack_dch_sqr = model.slack_es_dch[e, s_m, s_o, p] ** 2
                            obj += PENALTY_ESS * network.baseMVA * omega_market * omega_oper * (slack_ch_sqr + slack_dch_sqr)

            # Shared ESS slacks
            for e in model.shared_energy_storages:
                for p in model.periods:
                    if params.slacks.shared_ess.complementarity:
                        slack_comp = model.slack_shared_es_comp[e, s_m, s_o, p]
                        obj += PENALTY_SHARED_ESS * network.baseMVA * omega_market * omega_oper * slack_comp
                    if params.slacks.shared_ess.charging:
                        slack_ch_sqr = model.slack_shared_es_ch[e, s_m, s_o, p] ** 2
                        slack_dch_sqr = model.slack_shared_es_dch[e, s_m, s_o, p] ** 2
                        obj += PENALTY_SHARED_ESS * network.baseMVA * omega_market * omega_oper * (slack_ch_sqr + slack_dch_sqr)

    model.objective = pe.Objective(sense=pe.minimize, expr=obj)

    # Model suffixes (used for warm start)
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)  # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)  # Ipopt bound multipliers (sent to solver)
    model.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)  # Obtain dual solutions from previous solve and send to warm start

    return model


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


def _pre_process_network(network):

    processed_nodes = []
    for node in network.nodes:
        if node.type != BUS_ISOLATED:
            processed_nodes.append(node)

    processed_gens = []
    for gen in network.generators:
        node_type = network.get_node_type(gen.bus)
        if node_type != BUS_ISOLATED:
            processed_gens.append(gen)

    processed_branches = []
    for branch in network.branches:

        if not branch.is_connected():  # If branch is disconnected for all days and periods, remove
            continue

        if branch.pre_processed:
            continue

        fbus, tbus = branch.fbus, branch.tbus
        fnode_type = network.get_node_type(fbus)
        tnode_type = network.get_node_type(tbus)
        if fnode_type == BUS_ISOLATED or tnode_type == BUS_ISOLATED:
            branch.pre_processed = True
            continue

        parallel_branches = [branch for branch in network.branches if ((branch.fbus == fbus and branch.tbus == tbus) or (branch.fbus == tbus and branch.tbus == fbus))]
        connected_parallel_branches = [branch for branch in parallel_branches if branch.is_connected()]
        if len(connected_parallel_branches) > 1:
            processed_branch = connected_parallel_branches[0]
            r_eq, x_eq, g_eq, b_eq = _pre_process_parallel_branches(connected_parallel_branches)
            processed_branch.r = r_eq
            processed_branch.x = x_eq
            processed_branch.g_sh = g_eq
            processed_branch.b_sh = b_eq
            processed_branch.rate = sum([branch.rate for branch in connected_parallel_branches])
            processed_branch.ratio = branch.ratio
            processed_branch.pre_processed = True
            for branch_parallel in parallel_branches:
                branch_parallel.pre_processed = True
            processed_branches.append(processed_branch)
        else:
            for branch_parallel in parallel_branches:
                branch_parallel.pre_processed = True
            for branch_parallel in connected_parallel_branches:
                processed_branches.append(branch_parallel)

    network.nodes = processed_nodes
    network.generators = processed_gens
    network.branches = processed_branches
    for branch in network.branches:
        branch.pre_processed = False


def _pre_process_parallel_branches(branches):
    branch_impedances = [complex(branch.r, branch.x) for branch in branches]
    branch_shunt_admittance = [complex(branch.g_sh, branch.b_sh) for branch in branches]
    z_eq = 1/sum([(1/impedance) for impedance in branch_impedances])
    ysh_eq = sum([admittance for admittance in branch_shunt_admittance])
    return abs(z_eq.real), abs(z_eq.imag), ysh_eq.real, ysh_eq.imag
