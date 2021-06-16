# -*- coding: utf-8 -*-

__all__ = [
    'create_rng',
    'generate_validation_error',
    'get_file_extension'
]


###########
# IMPORTS #
###########

# Full

import numpy as np
import numpy.random as npr
import numpy.random.mtrand as nprm
import pathlib as pl

# Internal

from .custom_types import *
from .exceptions import *


#############
# FUNCTIONS #
#############

# noinspection PyProtectedMember
def create_rng(seed: oint) -> trand:

    if seed is None:
        return nprm._rand

    if isinstance(seed, (int, np.integer)):
        return npr.RandomState(int(seed))

    raise TypeError('The specified seed is not a valid RNG initializer.')


def generate_validation_error(e: texception, trace: tany) -> ValidationError:

    arguments = ''.join(trace[0][4]).split('=', 1)[0].strip()
    message = str(e).replace('@arg@', arguments)

    return ValidationError(message)


def get_file_extension(file_path: str) -> str:

    return ''.join(pl.Path(file_path).suffixes).lower()


def namedtuple_to_dictionary(obj: tany):

    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # noinspection PyProtectedMember
        return dict(zip(obj._fields, map(namedtuple_to_dictionary, obj)))

    if isinstance(obj, titerable) and not isinstance(obj, str):
        return type(obj)(map(namedtuple_to_dictionary, obj))

    if isinstance(obj, tmapping):
        return type(obj)(zip(obj.keys(), map(namedtuple_to_dictionary, obj.values())))

    return obj
