import os

import pandas as pd

transformator_type = {

    '110kW_free_breath',
    '220kW_protection',
    '220kW_free_breath',
}


def relative_concentration(a_t, accepted_value):
    return a_t / accepted_value


def get_accepted_maximum_value(name, year, transf_type=None):
    """
    Return accepted and maximum value
    :param name: ('H2', 'C2H2', 'C2H4', 'CO')
    :param year: operation year of transformator
    :param transf_type: ('35kW', '110kW_protection')
    :return: (accepted value, maximum value)
    """

    if name == 'H2':
        if transf_type == '35kW':
            if year < 5:
                return 0.005, 0.02
            if year >= 5:
                return 0.002, 0.01
        elif transf_type == '110kW_protection':
            if year < 5:
                return 0.006, 0.01
            if year >= 5:
                return 0.005, 0.009

    elif name == 'C2H2':
        if transf_type == '35kW':
            return 0.001, 0.0025
        elif transf_type == '110kW_protection':
            return 0.003, 0.0008

    elif name == 'C2H4':
        if transf_type == '35kW':
            return 0.003, 0.009
        elif transf_type == '110kW_protection':
            return 0.005, 0.01

    elif name == 'CO':
        if transf_type == '35kW':
            if year < 30:
                return 0.015, 0.03
            if year >= 30:
                return 0.018, 0.035
        elif transf_type == '110kW_protection':
            return 0.045, 0.07


def get_all_relative_concentration(data: pd.DataFrame):
    df_relative_concentration = pd.DataFrame(columns=['H2', 'CO', 'C2H4', 'C2H2'])
    for gas_name in ['H2', 'CO', 'C2H4', 'C2H2']:
        df_relative_concentration[gas_name] = data[gas_name].apply(
            lambda x: relative_concentration(x, get_accepted_maximum_value(gas_name, 0, '35kW')[0]))
    return df_relative_concentration


class Transformer:

    def __init__(self, transfromator_type: str, operation_year: int):
        self.type = transfromator_type
        self.operation_year = operation_year



