import os
import sys
import getopt
from operational_planning import OperationalPlanning


# ======================================================================================================================
#  Read Execution Arguments
# ======================================================================================================================
def print_help_message():
    print('\nShared Resources Planning Tool usage:')
    print('\n Usage: main.py [OPTIONS]')
    print('\n Options:')
    print('   {:25}  {}'.format('-d, --test_case=', 'Directory of the Test Case to be run (located inside "data" directory)'))
    print('   {:25}  {}'.format('-f, --specification_file=', 'Specification file of the Test Case to be run (located inside the "test_case" directory)'))
    print('   {:25}  {}'.format('-h, --help', 'Help. Displays this message'))


def read_execution_arguments(argv):

    test_case_dir = str()
    spec_filename = str()

    try:
        opts, args = getopt.getopt(argv, 'hd:f:', ['help', 'test_case=', 'specification_file='])

        if not argv or not opts:
            print_help_message()
            sys.exit()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_help_message()
                sys.exit()
            elif opt in ('-d', '--dir'):
                test_case_dir = arg
            elif opt in ('-f', '--file'):
                spec_filename = arg

    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)

    if not test_case_dir or not spec_filename:
        print('Tool usage:')
        print('\tmain.py -d <test_case> -f <specification_file>')
        sys.exit()

    return spec_filename, test_case_dir


# ======================================================================================================================
#  Main
# ======================================================================================================================
if __name__ == '__main__':

    filename, test_case = read_execution_arguments(sys.argv[1:])
    directory = os.path.join(os.getcwd(), 'data', test_case)

    operational_planning = OperationalPlanning(directory, filename)
    operational_planning.read_case_study()

    t = 11
    num_steps = 4

    # transmission_network = operational_planning.transmission_network
    # tn_model = transmission_network.build_model(t=11)
    # results = transmission_network.optimize(tn_model)
    # processed_results = transmission_network.process_results(tn_model, results)
    # transmission_network.write_optimization_results_to_excel(processed_results, filename='test')

    for _ in range(10):
        # operational_planning.run_without_coordination(t=t, filename=f'{operational_planning.name}_uncoordinated_t={t}')
        operational_planning.run_hierarchical_coordination(t=t, num_steps=num_steps, filename=f'{operational_planning.name}_hierarchical_t={t}', print_pq_map=False)
        # operational_planning.run_distributed_coordination(t=t, filename=f'{operational_planning.name}_distributed_t={t}')
        # operational_planning.run_distributed_coordination(t=t, consider_shared_ess=True, filename=f'{operational_planning.name}_distributed_ESS_t={t}')
