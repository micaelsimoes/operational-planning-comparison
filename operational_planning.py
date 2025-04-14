import os
import pandas as pd
from network import Network
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

    def read_case_study(self):
        _read_case_study(self)

    def read_market_data_from_file(self, filename):
        _read_market_data_from_file(self, filename)


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

        network_name = distribution_network_data['name']  # Network filename
        params_filename = distribution_network_data['params_filename']  # Params filename
        connection_nodeid = distribution_network_data['connection_node_id']  # Connection node ID

        distribution_network = Network()
        distribution_network.name = network_name
        distribution_network.is_transmission = False
        distribution_network.data_dir = operational_planning.data_dir
        distribution_network.results_dir = operational_planning.results_dir
        distribution_network.diagrams_dir = operational_planning.diagrams_dir
        distribution_network.cost_energy_p = operational_planning.cost_energy_p
        distribution_network.cost_flex = operational_planning.cost_flex
        distribution_network.params_file = params_filename
        distribution_network.read_network_parameters()
        distribution_network.read_network_data()


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
        value = data.iloc[i, 1]
        market_costs[cost_type] = value
    return market_costs
