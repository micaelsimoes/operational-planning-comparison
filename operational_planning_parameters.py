from helper_functions import *


# ======================================================================================================================
#  Class ADMM Parameters
# ======================================================================================================================
class ADMMParameters:

    def __init__(self):
        self.tol = {'consensus': {'v': 0.1e-2, 'pf': 0.1e-2, 'ess': 1e-2},
                    'stationarity': {'v': 0.5e-2, 'pf': 0.5e-2, 'ess': 5e-2}}
        self.num_max_iters = 1000
        self.adaptive_penalty = False
        self.rho = {'v': dict(), 'pf': dict(), 'ess': dict()}
        self.previous_iter = {'v': dict(), 'pf': dict(), 'ess': dict()}
        self.rho_previous_iter = {'v': dict(), 'pf': dict(), 'ess': dict()}

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(admm_params, filename):
    params_data = convert_json_to_dict(read_json_file(filename))
    admm_params.tol['consensus']['v'] = float(params_data['tol']['consensus']['v'])
    admm_params.tol['consensus']['pf'] = float(params_data['tol']['consensus']['pf'])
    admm_params.tol['consensus']['ess'] = float(params_data['tol']['consensus']['ess'])
    admm_params.tol['stationarity']['v'] = float(params_data['tol']['stationarity']['v'])
    admm_params.tol['stationarity']['pf'] = float(params_data['tol']['stationarity']['pf'])
    admm_params.tol['stationarity']['ess'] = float(params_data['tol']['stationarity']['ess'])
    admm_params.num_max_iters = int(params_data['num_max_iters'])
    admm_params.adaptive_penalty = bool(params_data['adaptive_penalty'])
    admm_params.rho['v'] = params_data['rho']['v']
    admm_params.rho['pf'] = params_data['rho']['pf']
    admm_params.rho['ess'] = params_data['rho']['ess']
    if 'v' in params_data['previous_iteration']:
        if bool(params_data['previous_iteration']['v']):
            print('[WARNING] Previous iteration interface voltage magnitude variables not implemented!')
    if 'pf' in params_data['previous_iteration']:
        if bool(params_data['previous_iteration']['pf']):
            print('[WARNING] Previous iteration interface power flow variables not implemented!')
    admm_params.previous_iter['ess']['tso'] = bool(params_data['previous_iteration']['ess']['tso'])
    admm_params.previous_iter['ess']['dso'] = bool(params_data['previous_iteration']['ess']['dso'])
    if admm_params.previous_iter['ess']['tso'] or admm_params.previous_iter['ess']['dso']:
        admm_params.rho_previous_iter['ess'] = params_data['rho_previous_iter']['ess']
