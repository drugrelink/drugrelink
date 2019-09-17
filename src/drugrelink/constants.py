# -*- coding: utf-8 -*-

"""Constants for repositioning comparison."""

import os

DEFAULT_GRAPH_TYPE = 'subgraph'
GRAPH_TYPES = [
    'graph',
    'subgraph',
    'permutation1',
    'permutation2',
    'permutation3',
    'permutation4',
    'permutation5',
]
HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'resources'))
RESOURCES_DIRECTORY = os.environ.get('DRUGRELINK_RESOURCES_DIRECTORY', DEFAULT_DIRECTORY)
