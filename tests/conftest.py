# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Partial

from os.path import (
    abspath,
    dirname,
    isfile,
    join
)

from json import (
    load
)


#########
# CACHE #
#########

fixtures = {}


#############
# FUNCTIONS #
#############

def sanitize_fixture_recursive(element, mapping):

    if isinstance(element, dict):
        return {key: sanitize_fixture_recursive(value, mapping) for key, value in element.items()}

    if isinstance(element, list):
        return [sanitize_fixture_recursive(item, mapping) for item in element]

    for m in mapping:
        if element == m[0]:
            return m[1]

    return element


def parse_fixture_dictionary(fixture, fixture_names, subtest_name):

    values = []
    ids = []

    expected_args = len(fixture_names)
    subtest_reference = f'{subtest_name.replace("test_", "")}_data'

    if subtest_reference in fixture:

        fixture_data = fixture[subtest_reference]

        if isinstance(fixture_data, dict):

            values_current = tuple([fixture_data[fixture_name] for fixture_name in fixture_names if fixture_name in fixture_data])

            if len(values_current) == expected_args:
                values.append(values_current)
                ids.append(f'{subtest_name}')

        elif isinstance(fixture_data, list):

            for index, case in enumerate(fixture_data):

                case_id = case['id'] if 'id' in case else f' #{str(index + 1)}'
                values_current = tuple([case[fixture_name] for fixture_name in fixture_names if fixture_name in case])

                if len(values_current) == expected_args:
                    values.append(values_current)
                    ids.append(f'{subtest_name}_{case_id}')

    return values, ids


def parse_fixture_list(fixture, fixture_names, subtest_name):

    values = []
    ids = []

    expected_args = len(fixture_names)
    subtest_reference = f'{subtest_name.replace("test_", "")}_data'

    if any([subtest_reference in case for case in fixture]):

        for index_case, case in enumerate(fixture):

            if subtest_reference in case:

                case_id = case['id'] if 'id' in case else f' #{str(index_case + 1)}'
                case_values = tuple([case[fixture_name] for fixture_name in fixture_names if fixture_name in case])

                for index_subcase, subcase in enumerate(case[subtest_reference]):

                    values_current = case_values + tuple([subcase[fixture_name] for fixture_name in fixture_names if fixture_name in subcase])

                    if len(values_current) == expected_args:
                        values.append(values_current)
                        ids.append(f'{subtest_name} {case_id}-{str(index_subcase + 1)}')

    else:

        for index, case in enumerate(fixture):

            case_id = case['id'] if 'id' in case else f' #{str(index + 1)}'
            values_current = tuple([case[fixture_name] for fixture_name in fixture_names if fixture_name in case])

            if len(values_current) == expected_args:
                values.append(values_current)
                ids.append(f'{subtest_name} {case_id}')

    return values, ids


#########
# SETUP #
#########

def pytest_configure(config):

    config.addinivalue_line('markers', 'slow: mark tests as slow (exclude them with \'-m "not slow"\').')


def pytest_generate_tests(metafunc):

    module = metafunc.module.__name__
    func = metafunc.definition.name
    mark = metafunc.definition.get_closest_marker('parametrize')
    names = metafunc.fixturenames

    test_index = module.find('_') + 1
    test_name = module[test_index:]

    if test_name not in fixtures:

        base_directory = abspath(dirname(__file__))
        fixtures_file = join(base_directory, f'fixtures/fixtures_{test_name}.json')

        if not isfile(fixtures_file):
            fixtures[test_name] = None
        else:

            with open(fixtures_file, 'r') as file:
                fixture = load(file)
                fixture = sanitize_fixture_recursive(fixture, [('NaN', float('nan')), ('Infinity', float('inf')), ('-Infinity', float('-inf'))])
                fixtures[test_name] = fixture

    fixture = fixtures[test_name]

    values = []
    ids = []

    if mark is None and fixture is not None and len(fixture) > 0:

        if isinstance(fixture, dict):
            values, ids = parse_fixture_dictionary(fixture, names, func)
        elif isinstance(fixture, list):
            values, ids = parse_fixture_list(fixture, names, func)

    metafunc.parametrize(names, values, False, ids)
