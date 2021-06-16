# -*- coding: utf-8 -*-

__all__ = [
    'read_csv',
    'read_json',
    'read_txt',
    'read_xml',
    'write_csv',
    'write_json',
    'write_txt',
    'write_xml'
]


###########
# IMPORTS #
###########

# Full

import numpy as np

# Partial

# noinspection PyPep8Naming
from csv import (
    reader as csv_reader,
    writer as csv_writer,
    QUOTE_MINIMAL as csv_quote_minimal
)

from io import (
    BytesIO
)

from json import (
    dump as json_dump,
    load as json_load
)

# noinspection PyPep8Naming
from xml.etree import (
    cElementTree as xml_tree
)

# Internal

from .custom_types import *


#############
# FUNCTIONS #
#############

def read_csv(file_path: str) -> tmc_dict:

    d = {}

    size = 0
    states = None

    with open(file_path, mode='r', newline='') as file:

        file.seek(0)

        if not file.read(1):  # pragma: no cover
            raise OSError('The file is empty.')
        else:
            file.seek(0)

        data = csv_reader(file)

        for index, row in enumerate(data):

            if index == 0:

                states = row

                if not all(isinstance(s, str) for s in states) or not all(s is not None and len(s) > 0 for s in states):  # pragma: no cover
                    raise ValueError('The file header is invalid.')

                size = len(states)
                states_unique = len(set(states))

                if states_unique < size:  # pragma: no cover
                    raise ValueError('The file header is invalid.')

            else:

                probabilities = row

                if len(probabilities) != size or not all(isinstance(p, str) for p in probabilities) or not all(p is not None and len(p) > 0 for p in probabilities):  # pragma: no cover
                    raise ValueError('The file contains invalid rows.')

                state_from = states[index - 1]

                for i in range(size):

                    state_to = states[i]

                    try:
                        probability = float(probabilities[i])
                    except Exception:  # pragma: no cover
                        raise ValueError('The file contains invalid rows.')

                    d[(state_from, state_to)] = probability

    return d


def read_json(file_path: str) -> tmc_dict:

    d = {}
    valid_keys = ['probability', 'state_from', 'state_to']

    with open(file_path, mode='r') as file:

        file.seek(0)

        if not file.read(1):  # pragma: no cover
            raise OSError('The file is empty.')
        else:
            file.seek(0)

        data = json_load(file)

        if not isinstance(data, list):  # pragma: no cover
            raise ValueError('The file format is not compliant.')

        for obj in data:

            if not isinstance(obj, dict):  # pragma: no cover
                raise ValueError('The file format is not compliant.')

            if sorted(list(obj.keys())) != valid_keys:  # pragma: no cover
                raise ValueError('The file contains invalid elements.')

            state_from = obj['state_from']
            state_to = obj['state_to']
            probability = obj['probability']

            if not isinstance(state_from, str) or len(state_from) == 0:  # pragma: no cover
                raise ValueError('The file contains invalid elements.')

            if not isinstance(state_to, str) or len(state_to) == 0:  # pragma: no cover
                raise ValueError('The file contains invalid elements.')

            if not isinstance(probability, (float, int, np.floating, np.integer)):  # pragma: no cover
                raise ValueError('The file contains invalid elements.')

            d[(state_from, state_to)] = float(probability)

    return d


def read_txt(file_path: str) -> tmc_dict:

    d = {}

    with open(file_path, mode='r') as file:

        file.seek(0)

        if not file.read(1):  # pragma: no cover
            raise OSError('The file is empty.')
        else:
            file.seek(0)

        for line in file:

            if not line.strip():  # pragma: no cover
                raise ValueError('The file contains invalid lines.')

            ls = line.split()

            if len(ls) != 3:  # pragma: no cover
                raise ValueError('The file contains invalid lines.')

            try:
                ls2 = float(ls[2])
            except Exception:  # pragma: no cover
                raise ValueError('The file contains invalid lines.')

            d[(ls[0], ls[1])] = ls2

    return d


def read_xml(file_path: str) -> tmc_dict:

    d = {}
    valid_keys = ['probability', 'state_from', 'state_to']

    with open(file_path, mode='r') as file:

        file.seek(0)

        if not file.read(1):  # pragma: no cover
            raise OSError('The file is empty.')
        else:
            file.seek(0)

    try:
        document = xml_tree.parse(file_path)
    except Exception:  # pragma: no cover
        raise ValueError('The file format is not compliant.')

    root = document.getroot()

    if root.tag != 'MarkovChain':  # pragma: no cover
        raise ValueError('The file root element is invalid.')

    for element in root.iter():

        if element.tag == 'MarkovChain':
            continue

        if element.tag != 'Transition':  # pragma: no cover
            raise ValueError('The file contains invalid subelements.')

        attributes = element.items()

        if len(attributes) == 0:  # pragma: no cover
            raise ValueError('The file contains invalid subelements.')

        keys = [attribute[0] for attribute in attributes]

        if sorted(keys) != valid_keys:  # pragma: no cover
            raise ValueError('The file contains invalid subelements.')

        values = [attribute[1].strip() for attribute in attributes]

        if any([len(value) == 0 for value in values]):  # pragma: no cover
            raise ValueError('The file contains invalid subelements.')

        index = keys.index('state_from')
        state_from = values[index]

        index = keys.index('state_to')
        state_to = values[index]

        index = keys.index('probability')
        probability = values[index]

        try:
            probability = float(probability)
        except Exception:  # pragma: no cover
            raise ValueError('The file contains invalid subelements.')

        d[(state_from, state_to)] = probability

    return d


def write_csv(d: tmc_dict, file_path: str):

    states = [key[0] for key in d.keys() if key[0] == key[1]]
    size = len(states)

    p = np.zeros((size, size), dtype=float)

    for it, ip in d.items():
        p[states.index(it[0]), states.index(it[1])] = ip

    with open(file_path, mode='w', newline='') as file:

        writer = csv_writer(file, delimiter=',', quoting=csv_quote_minimal, quotechar='"')

        writer.writerow(states)

        for i in range(size):
            row = [str(x) for x in p[i, :].tolist()]
            writer.writerow(row)


def write_json(d: tmc_dict, file_path: str):

    output = []

    for it, ip in d.items():
        output.append({'state_from': it[0], 'state_to': it[1], 'probability': ip})

    with open(file_path, mode='w') as file:
        json_dump(output, file)


def write_txt(d: tmc_dict, file_path: str):

    with open(file_path, mode='w') as file:

        for it, ip in d.items():
            file.write(f'{it[0]} {it[1]} {ip}\n')


def write_xml(d: tmc_dict, file_path: str):

    root = xml_tree.Element('MarkovChain')

    for it, ip in d.items():
        transition = xml_tree.SubElement(root, 'Transition')
        transition.set('state_from', it[0])
        transition.set('state_to', it[1])
        transition.set('probability', str(ip))

    document = xml_tree.ElementTree(root)

    with BytesIO() as buffer:
        document.write(buffer, 'utf-8', True)
        xml_content = str(buffer.getvalue(), 'utf-8')

    xml_content = xml_content.replace('?>', " standalone='yes' ?>")
    xml_content = xml_content.replace('<MarkovChain>', '<MarkovChain>\n')
    xml_content = xml_content.replace('<Transition', '\t<Transition')
    xml_content = xml_content.replace('" />', '"/>\n')

    with open(file_path, mode='w') as file:
        file.write(xml_content)
