import os
from energy_storage import EnergyStorage
from helper_functions import *


# ======================================================================================================================
#  SHARED ENERGY STORAGE Information
# ======================================================================================================================
class SharedEnergyStorageData:

    def __init__(self):
        self.data_dir = str()
        self.shared_energy_storages = list()

    def read_shared_energy_storage_data_from_file(self, shared_ess_filename):
        filename = os.path.join(self.data_dir, "Shared ESS", shared_ess_filename)
        _read_shared_energy_storage_data_from_file(self, filename)


def _read_shared_energy_storage_data_from_file(shared_energy_storage_data, filename):

    data = convert_json_to_dict(read_json_file(filename))

    for ess_data in data['shared_ess']:

        shared_ess = EnergyStorage()
        shared_ess.es_id = int(ess_data['es_id'])
        shared_ess.bus = int(ess_data['bus'])
        shared_ess.s = float(ess_data['s'])
        shared_ess.e = float(ess_data['e'])
        shared_ess.e_init = shared_ess.e * ENERGY_STORAGE_RELATIVE_INIT_SOC
        shared_ess.e_min = shared_ess.e * ENERGY_STORAGE_MIN_ENERGY_STORED
        shared_ess.e_max = shared_ess.e * ENERGY_STORAGE_MAX_ENERGY_STORED
        shared_ess.eff_ch = float(ess_data['eff_ch'])
        shared_ess.eff_dch = float(ess_data['eff_dch'])
        shared_ess.max_pf = float(ess_data['max_pf'])
        shared_ess.min_pf = float(ess_data['min_pf'])
        shared_energy_storage_data.shared_energy_storages.append(shared_ess)
