# -*- coding: utf-8 -*-

__all__ = [
    # Generic
    'ofloat', 'oint', 'ostr',
    'tany', 'texception', 'titerable', 'tmapping',
    'tarray', 'oarray',
    'tmc', 'omc',
    # Lists
    'tlist_any', 'olist_any',
    'tlist_array', 'olist_array',
    'tlist_float', 'olist_float',
    'tlist_int', 'olist_int',
    'tlist_str', 'olist_str',
    # Lists of Lists
    'tlists_any', 'olists_any',
    'tlists_array', 'olists_array',
    'tlists_float', 'olists_float',
    'tlists_int', 'olists_int',
    'tlists_str', 'olists_str',
    # Specific
    'tbcond', 'obcond',
    'tcache', 'ocache',
    'tdists_flex', 'odists_flex',
    'tgenres', 'ogenres',
    'tgenres_ext', 'ogenres_ext',
    'tgraph', 'ograph',
    'tgraphs', 'ographs',
    'tinterval', 'ointerval',
    'tlimit_float', 'olimit_float',
    'tlimit_int', 'olimit_int',
    'tmc_dict', 'omc_dict',
    'tmc_dict_flex', 'omc_dict_flex',
    'tmc_fit', 'omc_fit',
    'tnumeric', 'onumeric',
    'tpart', 'opart',
    'tparts', 'oparts',
    'tplot', 'oplot',
    'trand', 'orand',
    'trdl', 'ordl',
    'tstate', 'ostate',
    'tstates', 'ostates',
    'tstatus', 'ostatus',
    'ttfunc', 'otfunc',
    'ttimes_in', 'otimes_in',
    'ttimes_out', 'otimes_out',
    'twalk', 'owalk',
    'twalk_flex', 'owalk_flex',
    'tweights', 'oweights'
]


###########
# IMPORTS #
###########

# Full

import matplotlib.pyplot as pp
import networkx as nx
import numpy as np
import numpy.random as npr
import scipy.sparse as spsp

try:
    import pandas as pd
except ImportError:
    pd = None

# Partial

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


#########
# TYPES #
#########

# Generic

ofloat = Optional[float]
oint = Optional[int]
ostr = Optional[str]

tany = Any
texception = Exception
tmapping = Mapping
titerable = Iterable

tarray = np.ndarray
oarray = Optional[tarray]

tmc = TypeVar('MarkovChain')
omc = Optional[tmc]

# Lists

tlist_any = List[tany]
olist_any = Optional[tlist_any]

tlist_array = List[tarray]
olist_array = Optional[tlist_array]

tlist_float = List[float]
olist_float = Optional[tlist_float]

tlist_int = List[int]
olist_int = Optional[tlist_int]

tlist_str = List[str]
olist_str = Optional[tlist_str]

# Lists of Lists

tlists_any = List[tlist_any]
olists_any = Optional[tlists_any]

tlists_array = List[tlist_array]
olists_array = Optional[tlists_array]

tlists_float = List[tlist_float]
olists_float = Optional[tlists_float]

tlists_int = List[tlist_int]
olists_int = Optional[tlists_int]

tlists_str = List[tlist_str]
olists_str = Optional[tlists_str]

# Specific

tbcond = Union[float, str]
obcond = Optional[tbcond]

tcache = Dict[str, tany]
ocache = Optional[tcache]

tdists_flex = Union[int, tlist_array]
odists_flex = Optional[tdists_flex]

tgenres = Tuple[oarray, ostr]
ogenres = Optional[tgenres]

tgenres_ext = Tuple[oarray, olist_str, ostr]
ogenres_ext = Optional[tgenres_ext]

tgraph = nx.DiGraph
ograph = Optional[tgraph]

tgraphs = Union[tgraph, nx.MultiDiGraph]
ographs = Optional[tgraphs]

tinterval = Tuple[Union[float, int], Union[float, int]]
ointerval = Optional[tinterval]

tlimit_float = Tuple[float, bool]
olimit_float = Optional[tlimit_float]

tlimit_int = Tuple[int, bool]
olimit_int = Optional[tlimit_int]

tmc_dict = Dict[Tuple[str, str], float]
omc_dict = Optional[tmc_dict]

tmc_dict_flex = Dict[Tuple[str, str], Union[float, int]]
omc_dict_flex = Optional[tmc_dict_flex]

tmc_fit = Tuple[tmc, tlist_array]
omc_fit = Optional[tmc_fit]

tnumeric = Union[titerable, tarray, spsp.spmatrix, pd.DataFrame, pd.Series] if pd is not None else Union[titerable, tarray, spsp.spmatrix]
onumeric = Optional[tnumeric]

tpart = List[Union[tlist_int, tlist_str]]
opart = Optional[tpart]

tparts = List[tpart]
oparts = Optional[tparts]

tplot = Tuple[pp.Figure, pp.Axes]
oplot = Optional[tplot]

trand = npr.RandomState
orand = Optional[trand]

trdl = Tuple[tarray, tarray, tarray]
ordl = Optional[trdl]

tstate = Union[int, str]
ostate = Optional[tstate]

tstates = Union[tstate, tlist_int, tlist_str]
ostates = Optional[tstates]

tstatus = Union[int, str, tnumeric]
ostatus = Optional[tstatus]

ttfunc = Callable[[float, float], float]
otfunc = Optional[ttfunc]

ttimes_in = Union[int, tlist_int]
otimes_in = Optional[ttimes_in]

ttimes_out = Union[float, tlist_float]
otimes_out = Optional[ttimes_out]

twalk = Union[tlist_int, tlist_str]
owalk = Optional[twalk]

twalk_flex = Union[int, twalk]
owalk_flex = Optional[twalk_flex]

tweights = Union[float, int, tnumeric]
oweights = Optional[tweights]
