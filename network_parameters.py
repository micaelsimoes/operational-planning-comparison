from helper_functions import *
from solver_parameters import SolverParameters


# ======================================================================================================================
#   Class NetworkParameters
# ======================================================================================================================
class NetworkParameters:

    def __init__(self):
        self.obj_type = OBJ_MIN_COST
        self.transf_reg = True
        self.es_reg = True
        self.fl_reg = True
        self.rg_curt = False
        self.l_curt = False
        self.enforce_vg = False
        self.branch_limit_type = BRANCH_LIMIT_CURRENT
        self.slacks = Slacks()
        self.print_to_screen = False
        self.plot_diagram = False
        self.print_results_to_file = False
        self.solver_params = SolverParameters()

    def read_parameters_from_file(self, filename):
        _read_network_parameters_from_file(self, filename)


# ======================================================================================================================
#   Slack Classes
# ======================================================================================================================
class Slacks:

    def __init__(self):
        self.grid_operation = SlacksOperation()
        self.ess = SlacksEnergyStorage()
        self.shared_ess = SlacksEnergyStorage()
        self.node_balance = False

    def read_slacks_parameters(self, slacks_data):
        self.grid_operation.read_slacks_parameters(slacks_data)
        self.ess.read_slacks_parameters(slacks_data)
        self.shared_ess.read_slacks_parameters(slacks_data)
        self.node_balance = bool(slacks_data["node_balance"])


class SlacksOperation:

    def __init__(self):
        self.voltage = False
        self.branch_flow = False

    def read_slacks_parameters(self, slacks_data):
        if 'grid_operation' in slacks_data:
            if 'voltage' in slacks_data['grid_operation']:
                self.voltage = slacks_data['grid_operation']['voltage']
            if 'branch_flow' in slacks_data['grid_operation']:
                self.branch_flow = slacks_data['grid_operation']['branch_flow']


class SlacksEnergyStorage:

    def __init__(self):
        self.complementarity = False
        self.charging = False

    def read_slacks_parameters(self, slacks_data):
        if 'ess' in slacks_data:
            _read_ess_slacks_parameters(self, slacks_data['ess'])
        elif 'shared_ess' in slacks_data:
            _read_ess_slacks_parameters(self, slacks_data['shared_ess'])


# ======================================================================================================================
#   Read functions
# ======================================================================================================================
def _read_network_parameters_from_file(parameters, filename):

    params_data = convert_json_to_dict(read_json_file(filename))

    if params_data['obj_type'] == 'COST':
        parameters.obj_type = OBJ_MIN_COST
    elif params_data['obj_type'] == 'CONGESTION_MANAGEMENT':
        parameters.obj_type = OBJ_CONGESTION_MANAGEMENT
    else:
        print('[ERROR] Invalid objective type. Exiting...')
        exit(ERROR_PARAMS_FILE)
    parameters.transf_reg = bool(params_data['transf_reg'])
    parameters.es_reg = bool(params_data['es_reg'])
    parameters.fl_reg = bool(params_data['fl_reg'])
    parameters.rg_curt = bool(params_data['rg_curt'])
    parameters.l_curt = bool(params_data['l_curt'])
    parameters.enforce_vg = bool(params_data['enforce_vg'])
    if 'branch_limit_type' in params_data:
        if params_data['branch_limit_type'] == 'CURRENT':
            parameters.branch_limit_type = BRANCH_LIMIT_CURRENT
        elif params_data['branch_limit_type'] == 'APPARENT_POWER':
            parameters.branch_limit_type = BRANCH_LIMIT_APPARENT_POWER
        elif params_data['branch_limit_type'] == 'MIXED':
            parameters.branch_limit_type = BRANCH_LIMIT_MIXED
        else:
            print('[ERROR] Invalid objective type. Exiting...')
            exit(ERROR_PARAMS_FILE)
    if 'slacks' in params_data:
        parameters.slacks.read_slacks_parameters(params_data['slacks'])
    parameters.solver_params.read_solver_parameters(params_data['solver'])
    parameters.print_to_screen = params_data['print_to_screen']
    parameters.plot_diagram = params_data['plot_diagram']
    parameters.print_results_to_file = params_data['print_results_to_file']


def _read_ess_slacks_parameters(ess_data, slacks_data):
    if 'complementarity' in slacks_data:
        ess_data.complementarity = slacks_data['complementarity']
    if 'charging' in slacks_data:
        ess_data.charging = slacks_data['charging']
