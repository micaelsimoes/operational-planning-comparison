import sys
import json
from definitions import *


def read_json_file(filename):
    try:
        input_file = open(filename, 'r')
        with input_file:
            file_contents = input_file.read()
            return file_contents
    except OSError:
        print(f'[ERROR] Could not open file {filename}. Exiting...')
        sys.exit()


def convert_json_to_dict(json_string):
    try:
        data_dict = json.loads(json_string)
        return data_dict
    except json.JSONDecodeError as e:
        print(f'[ERROR] Could not convert JSON to dict. JSONDecodeError: {e}')
        exit(ERROR_SPECIFICATION_FILE)
    except TypeError as e:
        print(f'[ERROR] Could not convert JSON to dict. TypeError: {e}')
        exit(ERROR_SPECIFICATION_FILE)


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
