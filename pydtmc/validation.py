# -*- coding: utf-8 -*-

__all__ = [
    'validate_boolean',
    'validate_boundary_condition',
    'validate_dictionary',
    'validate_distribution',
    'validate_dpi',
    'validate_enumerator',
    'validate_float',
    'validate_graph',
    'validate_hyperparameter',
    'validate_integer',
    'validate_interval',
    'validate_markov_chain',
    'validate_mask',
    'validate_matrix',
    'validate_partitions',
    'validate_rewards',
    'validate_state',
    'validate_state_names',
    'validate_states',
    'validate_status',
    'validate_string',
    'validate_time_points',
    'validate_transition_function',
    'validate_transition_matrix',
    'validate_vector'
]


###########
# IMPORTS #
###########

# Full

import networkx as nx
import numpy as np
import scipy.sparse as spsp

try:
    import pandas as pd
except ImportError:
    pd = None

# Partial

from copy import (
    deepcopy
)

from inspect import (
    signature
)

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############

def extract(data: tany) -> tlist_any:

    result = None

    if isinstance(data, list):
        result = deepcopy(data)
    elif isinstance(data, dict):
        result = list(data.values())
    elif isinstance(data, titerable) and not isinstance(data, str):
        result = list(data)

    if result is None:
        raise TypeError('The data type is not supported.')

    return result


def extract_as_numeric(data: tany) -> tarray:

    result = None

    if isinstance(data, list):
        result = np.array(data)
    elif isinstance(data, dict):
        result = np.array(list(data.values()))
    elif isinstance(data, np.ndarray):
        result = np.copy(data)
    elif isinstance(data, spsp.spmatrix):
        result = np.array(data.todense())
    elif pd is not None and isinstance(data, (pd.DataFrame, pd.Series)):
        result = data.to_numpy(copy=True)
    elif isinstance(data, titerable) and not isinstance(data, str):
        result = np.array(list(data))

    if result is None or not np.issubdtype(result.dtype, np.number):
        raise TypeError('The data type is not supported.')

    return result


def validate_boolean(value: tany) -> bool:

    if isinstance(value, bool):
        return value

    raise TypeError('The "@arg@" parameter must be a boolean value.')


def validate_boundary_condition(boundary_condition: tany) -> tbcond:

    if isinstance(boundary_condition, (float, np.floating)):

        boundary_condition = float(boundary_condition)

        if (boundary_condition < 0.0) or (boundary_condition > 1.0):
            raise ValueError('The "@arg@" parameter, when specified as a float, must have a value between 0 and 1.')

        return boundary_condition

    if isinstance(boundary_condition, str):

        possible_values = ['absorbing', 'reflecting']

        if boundary_condition not in possible_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must have one of the following values: {", ".join(possible_values)}.')

        return boundary_condition

    raise TypeError('The "@arg@" parameter must be either a float representing the first probability of the semi-reflecting condition or a string representing the boundary condition type.')


def validate_dictionary(d: tany) -> tmc_dict:

    if not isinstance(d, dict):
        raise ValueError('The "@arg@" parameter must be a dictionary.')

    keys = d.keys()

    if not all(isinstance(key, tuple) and (len(key) == 2) and isinstance(key[0], str) and isinstance(key[1], str) for key in keys):
        raise ValueError('The "@arg@" parameter keys must be tuples containing 2 string values.')

    keys = [key[0] for key in keys if key[0] == key[1]]

    if not all(key is not None and (len(key) > 0) for key in keys):
        raise TypeError('The "@arg@" parameter keys must contain only valid string values.')

    values = d.values()

    if not all(isinstance(value, (float, int, np.floating, np.integer)) for value in values):
        raise ValueError('The "@arg@" parameter values must be float or integer numbers.')

    values = [float(value) for value in values]

    if not all((value >= 0.0) and (value <= 1.0) for value in values):
        raise ValueError('The "@arg@" parameter values can contain only numbers between 0 and 1.')

    dictionary = {}

    for key, value in d.items():
        dictionary[key] = float(value)

    return dictionary


def validate_distribution(distribution: tany, size: int) -> tdists_flex:

    if isinstance(distribution, (int, np.integer)):

        distribution = int(distribution)

        if distribution <= 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 1.')

        return distribution

    elif isinstance(distribution, list):

        distribution_len = len(distribution)

        if distribution_len == 0:
            raise ValueError('The "@arg@" parameter, when specified as a list of vectors, must be non-empty.')

        for index, vector in enumerate(distribution):

            if not isinstance(vector, np.ndarray) or not np.issubdtype(vector.dtype, np.number):
                raise TypeError('The "@arg@" parameter must contain only numeric vectors.')

            if distribution_len <= 1:
                raise ValueError('The "@arg@" parameter, when specified as a list of vectors, must contain at least 2 elements.')

            vector = vector.astype(float)
            distribution[index] = vector

            if (vector.ndim != 1) or (vector.size != size):
                raise ValueError('The "@arg@" parameter must contain only vectors of size {size:d}.')

            if not all(np.isfinite(x) and (x >= 0.0) and (x <= 1.0) for x in np.nditer(vector)):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of values between 0 and 1.')

            if not np.isclose(np.sum(vector), 1.0):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of values whose sum is 1.')

        return distribution

    else:
        raise TypeError('The "@arg@" parameter must be either an integer representing the number of redistributions to perform or a list of valid distributions.')


def validate_dpi(value: tany) -> int:

    if not isinstance(value, (int, np.integer)):
        raise TypeError('The "@arg@" parameter must be an integer value.')

    value = int(value)

    if value not in [75, 100, 150, 200, 300]:
        raise ValueError('The "@arg@" parameter must have one of the following values: 75, 100, 150, 200, 300.')

    return value


def validate_enumerator(value: tany, possible_values: tlist_str) -> str:

    if not isinstance(value, str):
        raise TypeError('The "@arg@" parameter must be a string value.')

    if value not in possible_values:
        raise ValueError(f'The "@arg@" parameter value must be one of the following: {", ".join(possible_values)}.')

    return value


def validate_float(value: tany, lower_limit: olimit_float = None, upper_limit: olimit_float = None) -> float:

    if not isinstance(value, (float, np.floating)):
        raise TypeError('The "@arg@" parameter must be a float value.')

    value = float(value)

    if not np.isfinite(value) or not np.isreal(value):
        raise ValueError('The "@arg@" parameter be a real finite float value.')

    if lower_limit is not None:
        if lower_limit[1]:
            if value <= lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than {lower_limit[0]:f}.')
        else:
            if value < lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than or equal to {lower_limit[0]:f}.')

    if upper_limit is not None:
        if upper_limit[1]:
            if value >= upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than {upper_limit[0]:f}.')
        else:
            if value > upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than or equal to {upper_limit[0]:f}.')

    return value


def validate_graph(graph: tany) -> tgraphs:

    if graph is None:
        raise ValueError('The "@arg@" parameter must be a directed graph.')

    non_multi = isinstance(graph, nx.DiGraph)
    multi = isinstance(graph, nx.MultiDiGraph)

    if not non_multi and not multi:
        raise ValueError('The "@arg@" parameter must be a directed graph.')

    if multi:
        graph = nx.DiGraph(graph)

    size = len(list(graph.nodes))

    if size < 2:
        raise ValueError('The "@arg@" parameter must contain a number of nodes greater than or equal to 2.')

    return graph


def validate_hyperparameter(hyperparameter: tany, size: int) -> tarray:

    try:
        hyperparameter = extract_as_numeric(hyperparameter)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not np.issubdtype(hyperparameter.dtype, np.number):
        raise TypeError('The "@arg@" parameter must contain only integer values.')

    hyperparameter = hyperparameter.astype(float)

    if (hyperparameter.ndim != 2) or (hyperparameter.shape[0] != hyperparameter.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix.')

    if hyperparameter.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter size must be equal to {size:d}.')

    if not all(np.isfinite(x) and np.isreal(x) and np.equal(np.mod(x, 1), 0) and (x >= 1.0) for x in np.nditer(hyperparameter)):
        raise ValueError('The "@arg@" parameter must contain only integer values greater than or equal to 1.')

    return hyperparameter


def validate_integer(value: tany, lower_limit: olimit_int = None, upper_limit: olimit_int = None) -> int:

    if not isinstance(value, (int, np.integer)):
        raise TypeError('The "@arg@" parameter must be an integer value.')

    value = int(value)

    if lower_limit is not None:
        if lower_limit[1]:
            if value <= lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than {lower_limit[0]:d}.')
        else:
            if value < lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than or equal to {lower_limit[0]:d}.')

    if upper_limit is not None:
        if upper_limit[1]:
            if value >= upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than {upper_limit[0]:d}.')
        else:
            if value > upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than or equal to {upper_limit[0]:d}.')

    return value


def validate_interval(interval: tany) -> tinterval:

    if not isinstance(interval, tuple):
        raise TypeError('The "@arg@" parameter must be a tuple.')

    if len(interval) != 2:
        raise ValueError('The "@arg@" parameter must contain 2 elements.')

    a = interval[0]
    b = interval[1]

    if not isinstance(a, (float, int, np.floating, np.integer)) or not isinstance(b, (float, int, np.floating, np.integer)):
        raise ValueError('The "@arg@" parameter must contain only float and integer values.')

    a = float(a)
    b = float(b)

    if not all(np.isfinite(x) and np.isreal(x) for x in [a, b]):
        raise ValueError('The "@arg@" parameter must contain only real finite float values and integer values.')

    if a >= b:
        raise ValueError('The "@arg@" parameter must two distinct values, and the first value must be less than the second one.')

    return a, b


def validate_markov_chain(mc: tany) -> tmc:

    if mc is None or (type(mc).__name__ != 'MarkovChain'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    return mc


def validate_mask(mask: tany, size: int) -> tarray:

    try:
        mask = extract_as_numeric(mask)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    mask = mask.astype(float)

    if (mask.ndim != 2) or (mask.shape[0] != mask.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix.')

    if mask.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter size must be equal to {size:d}.')

    if not all(np.isnan(x) or ((x >= 0.0) and (x <= 1.0)) for x in np.nditer(mask)):
        raise ValueError('The "@arg@" parameter can contain only NaNs and values between 0 and 1.')

    if np.any(np.nansum(mask, axis=1, dtype=float) > 1.0):
        raise ValueError('The "@arg@" parameter row sums must not exceed 1.')

    return mask


def validate_matrix(m: tany) -> tarray:

    try:
        m = extract_as_numeric(m)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not np.issubdtype(m.dtype, np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    m = m.astype(float)

    if (m.ndim != 2) or (m.shape[0] < 2) or (m.shape[0] != m.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(np.isfinite(x) for x in np.nditer(m)):
        raise ValueError('The "@arg@" parameter must contain only finite values.')

    return m


def validate_partitions(partitions: tany, current_states: tlist_str) -> tlists_int:

    if not isinstance(partitions, list):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    partitions_length = len(partitions)
    current_states_length = len(current_states)

    if (partitions_length < 2) or (partitions_length >= current_states_length):
        raise ValueError(f'The "@arg@" parameter must contain a number of elements between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    partitions_flat = []
    partitions_groups = []

    for partition in partitions:

        if not isinstance(partition, titerable):
            raise TypeError('The "@arg@" parameter must contain only array_like objects.')

        partition_list = list(partition)

        partitions_flat.extend(partition_list)
        partitions_groups.append(len(partition_list))

    if all(isinstance(state, (int, np.integer)) for state in partitions_flat):

        partitions_flat = [int(state) for state in partitions_flat]

        if any((state < 0) or (state >= current_states_length) for state in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as integers, must be values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    elif all(isinstance(s, str) for s in partitions_flat):

        partitions_flat = [current_states.index(s) if s in current_states else -1 for s in partitions_flat]

        if any(s == -1 for s in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

    else:
        raise TypeError('The "@arg@" parameter must contain only array_like objects of integers or array_like objects of strings.')

    partitions_flat_length = len(partitions_flat)

    if len(set(partitions_flat)) < partitions_flat_length:
        raise ValueError('The "@arg@" parameter subelements must be unique.')

    if partitions_flat_length != current_states_length:
        raise ValueError('The "@arg@" parameter subelements must include all the existing states.')

    if partitions_flat != list(range(current_states_length)):
        raise ValueError('The "@arg@" parameter subelements must follow a sequential order.')

    partitions = []
    partitions_offset = 0

    for partitions_group in partitions_groups:
        partitions_extension = partitions_offset + partitions_group
        partitions.append(partitions_flat[partitions_offset:partitions_extension])
        partitions_offset += partitions_group

    return partitions


def validate_rewards(rewards: tany, size: int) -> tarray:

    try:
        rewards = extract_as_numeric(rewards)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not np.issubdtype(rewards.dtype, np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    rewards = rewards.astype(float)

    if (rewards.ndim < 1) or ((rewards.ndim == 2) and (rewards.shape[0] != 1)) or (rewards.ndim > 2):
        raise ValueError('The "@arg@" parameter must be a vector.')

    rewards = np.ravel(rewards)

    if rewards.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) for x in np.nditer(rewards)):
        raise ValueError('The "@arg@" parameter must contain only real finite values.')

    return rewards


def validate_state(state: tany, current_states: list) -> int:

    if isinstance(state, (int, np.integer)):

        state = int(state)
        limit = len(current_states) - 1

        if (state < 0) or (state > limit):
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        return state

    if isinstance(state, str):

        if state not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        return current_states.index(state)

    raise TypeError('The "@arg@" parameter must be either an integer or a string.')


def validate_status(status: tany, current_states: list) -> tarray:

    size = len(current_states)

    if isinstance(status, (int, np.integer)):

        status = int(status)
        limit = size - 1

        if (status < 0) or (status > limit):
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        result = np.zeros(size, dtype=float)
        result[status] = 1.0

        return result

    if isinstance(status, str):

        if status not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        status = current_states.index(status)

        result = np.zeros(size, dtype=float)
        result[status] = 1.0

        return result

    try:
        status = extract_as_numeric(status)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not np.issubdtype(status.dtype, np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    status = status.astype(float)

    if (status.ndim < 1) or ((status.ndim == 2) and (status.shape[0] != 1) and (status.shape[1] != 1)) or (status.ndim > 2):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    status = np.ravel(status)

    if status.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and (x >= 0.0) and (x <= 1.0) for x in np.nditer(status)):
        raise ValueError('The "@arg@" parameter must contain only values between 0 and 1.')

    if not np.isclose(np.sum(status), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return status


def validate_state_names(states: tany, size: oint = None) -> tlist_str:

    try:
        states = extract(states)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not all(isinstance(s, str) for s in states):
        raise TypeError('The "@arg@" parameter must contain only string values.')

    if not all(s is not None and (len(s) > 0) for s in states):
        raise TypeError('The "@arg@" parameter must contain only valid string values.')

    states_length = len(states)
    states_unique = len(set(states))

    if states_unique < states_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if size is not None and (states_length != size):
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    return states


def validate_states(states: tany, current_states: tlist_str, state_type: str, flex: bool) -> tlist_int:

    if flex:

        if isinstance(states, (int, np.integer)):

            states = int(states)
            limit = len(current_states) - 1

            if (states < 0) or (states > limit):
                raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

            return [states]

        if (state_type != 'walk') and isinstance(states, str):

            if states not in current_states:
                raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

            return [current_states.index(states)]

    try:
        states = extract(states)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    current_states_length = len(current_states)

    if all(isinstance(s, (int, np.integer)) for s in states):

        states = [int(state) for state in states]

        if any((s < 0) or (s >= current_states_length) for s in states):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    elif all(isinstance(s, str) for s in states):

        states = [current_states.index(s) if s in current_states else -1 for s in states]

        if any(s == -1 for s in states):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

    else:

        if flex:
            if state_type == 'walk':
                raise TypeError('The "@arg@" parameter must be either an integer, an array_like object of integers or an array_like object of strings.')
            else:
                raise TypeError('The "@arg@" parameter must be either an integer, a string, an array_like object of integers or an array_like object of strings.')
        else:
            raise TypeError('The "@arg@" parameter must be either an array_like object of integers or an array_like object of strings.')

    states_length = len(states)

    if (state_type != 'walk') and (len(set(states)) < states_length):
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if state_type == 'regular':

        if (states_length < 1) or (states_length > current_states_length):
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states ({current_states_length:d}).')

        states = sorted(states)

    elif state_type == 'subset':

        if (states_length < 1) or (states_length >= current_states_length):
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states minus one ({current_states_length - 1:d}).')

        states = sorted(states)

    else:

        if states_length < 2:
            raise ValueError('The "@arg@" parameter must contain at least two elements.')

    return states


def validate_string(value: tany) -> str:

    if not isinstance(value, str):
        raise TypeError('The "@arg@" parameter must be a string value.')

    value = value.strip()

    if len(value) == 0:
        raise ValueError('The "@arg@" parameter must not be a non-empty string.')

    return value


def validate_time_points(time_points: tany) -> ttimes_in:

    if isinstance(time_points, (int, np.integer)):

        time_points = int(time_points)

        if time_points < 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 0.')

        return time_points

    try:
        time_points = extract(time_points)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if all(isinstance(time_point, (int, np.integer)) for time_point in time_points):

        time_points = [int(time_point) for time_point in time_points]

        if any(time_point < 0 for time_point in time_points):
            raise ValueError('The "@arg@" parameter, when specified as a list of integers, must contain only values greater than or equal to 0.')

    else:
        raise TypeError('The "@arg@" parameter must be either an integer or an array_like object of integers.')

    time_points_length = len(time_points)

    if time_points_length < 1:
        raise ValueError('The "@arg@" parameter must contain at least one element.')

    if len(set(time_points)) < time_points_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    time_points = sorted(time_points)

    return time_points


def validate_transition_function(f: tany) -> ttfunc:

    if not callable(f):
        raise TypeError('The "@arg@" parameter must be a callable.')

    s = signature(f)

    if len(s.parameters) != 2:
        raise ValueError('The "@arg@" parameter must accept 2 input arguments.')

    return f


def validate_transition_matrix(p: tany) -> tarray:

    try:
        p = extract_as_numeric(p)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not np.issubdtype(p.dtype, np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    p = p.astype(float)

    if (p.ndim != 2) or (p.shape[0] != p.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix.')

    size = p.shape[0]

    if size < 2:
        raise ValueError('The "@arg@" parameter size must be greater than or equal to 2.')

    if not all(np.isfinite(x) and (x >= 0.0) and (x <= 1.0) for x in np.nditer(p)):
        raise ValueError('The "@arg@" parameter must contain only values between 0 and 1.')

    if not np.allclose(np.sum(p, axis=1), np.ones(size, dtype=float)):
        raise ValueError('The "@arg@" parameter rows must sum to 1.')

    return p


def validate_transition_matrix_size(size: tany) -> int:

    if not isinstance(size, (int, np.integer)):
        raise TypeError('The "@arg@" parameter must be an integer value.')

    size = int(size)

    if size < 2:
        raise ValueError('The "@arg@" parameter must be greater than or equal to 2.')

    return size


def validate_vector(vector: tany, vector_type: str, flex: bool, size: oint = None) -> tarray:

    if flex and size is not None and isinstance(vector, (float, int, np.floating, np.integer)):
        vector = np.repeat(float(vector), size)
    else:
        try:
            vector = extract_as_numeric(vector)
        except Exception:
            raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not np.issubdtype(vector.dtype, np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    vector = vector.astype(float)

    if (vector.ndim < 1) or ((vector.ndim == 2) and (vector.shape[0] != 1) and (vector.shape[1] != 1)) or (vector.ndim > 2):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    vector = np.ravel(vector)

    if size is not None and (vector.size != size):
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) and (x >= 0.0) and (x <= 1.0) for x in np.nditer(vector)):
        raise ValueError('The "@arg@" parameter must contain only values between 0 and 1.')

    if vector_type == 'annihilation':
        if not np.isclose(vector[0], 0.0):
            raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the first index.')
    elif vector_type == 'creation':
        if not np.isclose(vector[-1], 0.0):
            raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the last index.')
    elif vector_type == 'stochastic':
        if not np.isclose(np.sum(vector), 1.0):
            raise ValueError('The "@arg@" parameter values must sum to 1.')

    return vector
