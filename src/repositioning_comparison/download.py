# -*- coding: utf-8 -*-

"""Helper functions for getting resources."""

import logging
import os
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

HERE = os.path.abspath(os.path.dirname(__file__))

DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'data'))
DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)

NODE_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/nodes.tsv'
EDGE_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/edges.sif.gz'
TRANSFORMED_FEATURES_URL = 'https://github.com/dhimmel/learn/blob/master/prediction/features/transformed-features.tsv.bz2?raw=true'
VALIDATE_DATA_URL = 'https://github.com/dhimmel/learn/blob/master/validate/validation-statuses.tsv'
PERMUTATION1_DATA_URL = 'https://github.com/dhimmel/integrate/blob/master/data/permuted/hetnet_perm-1.json.bz2'
PERMUTATION2_DATA_URL = 'https://github.com/dhimmel/integrate/blob/master/data/permuted/hetnet_perm-2.json.bz2'
PERMUTATION3_DATA_URL = 'https://github.com/dhimmel/integrate/blob/master/data/permuted/hetnet_perm-3.json.bz2'
PERMUTATION4_DATA_URL = 'https://github.com/dhimmel/integrate/blob/master/data/permuted/hetnet_perm-4.json.bz2'
PERMUTATION5_DATA_URL = 'https://github.com/dhimmel/integrate/blob/master/data/permuted/hetnet_perm-5.json.bz2'

PERMUTATION_DATA_FILE_FMT = 'hetnet_perm-{}.json.bz2'
PERMUTATION_DATA_URL_FMT = 'https://github.com/dhimmel/integrate/blob/master/data/permuted/hetnet_perm-{}.json.bz2'


def ensure_data(directory: Optional[str] = None) -> Tuple[str, str, str, str, List[str]]:
    """Ensure Himmelstein's data files are downloaded."""
    if directory is None:
        directory = DIRECTORY

    os.makedirs(directory, exist_ok=True)

    node_data_path = os.path.join(directory, 'nodes.tsv')
    if not os.path.exists(node_data_path):
        logger.warning(f'downloading {NODE_DATA_URL}')
        urlretrieve(NODE_DATA_URL, node_data_path)

    edge_data_path = os.path.join(directory, 'edges.sif.gz')
    if not os.path.exists(edge_data_path):
        logger.warning(f'downloading {EDGE_DATA_URL}')
        urlretrieve(EDGE_DATA_URL, edge_data_path)

    transformed_features_path = os.path.join(directory, 'transformed-features.tsv.bz2')
    if not os.path.exists(transformed_features_path):
        logger.warning(f'downloading {TRANSFORMED_FEATURES_URL}')
        urlretrieve(TRANSFORMED_FEATURES_URL, transformed_features_path)

    validate_data_path = os.path.join(directory, 'validation-statuses.tsv')
    if not os.path.exists(validate_data_path):
        logger.warning(f'downloading {VALIDATE_DATA_URL}')
        urlretrieve(VALIDATE_DATA_URL, validate_data_path)

    permutation_directory = os.path.join(directory, "permutations")
    os.makedirs(permutation_directory, exist_ok=True)

    permutation_paths = []
    for i in range(5):
        permutation_data_path = os.path.join(permutation_directory, PERMUTATION_DATA_FILE_FMT.format(i + 1))
        if not os.path.exists(permutation_data_path):
            url = PERMUTATION_DATA_URL_FMT.format(i + 1)
            logger.warning(f'downloading {url}')
            urlretrieve(url, permutation_data_path)
        permutation_paths.append(permutation_data_path)

    return node_data_path, edge_data_path, transformed_features_path, validate_data_path, permutation_paths
