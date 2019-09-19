# -*- coding: utf-8 -*-

"""Helper functions for getting resources."""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

HERE = os.path.abspath(os.path.dirname(__file__))

DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'data'))
DATA_DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)

# URLs from dhimmel/integrate

NODE_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/nodes.tsv'
EDGE_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/edges.sif.gz'

PERMUTATION1_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/permuted/hetnet_perm-1.json.bz2'
PERMUTATION2_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/permuted/hetnet_perm-2.json.bz2'
PERMUTATION3_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/permuted/hetnet_perm-3.json.bz2'
PERMUTATION4_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/permuted/hetnet_perm-4.json.bz2'
PERMUTATION5_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/permuted/hetnet_perm-5.json.bz2'

PERMUTATION_DATA_FILE_FMT = 'hetnet_perm-{}.json.bz2'
PERMUTATION_DATA_URL_FMT = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/permuted/hetnet_perm-{}.json.bz2'

# URLs from dhimmel/learn

TRANSFORMED_FEATURES_URL = 'https://raw.githubusercontent.com/dhimmel/learn/master/prediction/features/transformed-features.tsv.bz2?raw=true'
VALIDATE_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/learn/master/validate/validation-statuses.tsv'
SYMPTOMATIC_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/learn/master/prediction/predictions/probabilities.tsv'


@dataclass
class DataPaths:
    """Container for the paths for training."""

    node_data_path: str
    edge_data_path: str
    transformed_features_path: str
    validate_data_path: str
    symptomatic_data_path: str
    permutation_paths: List[str]
    data_edge2vec_path: str


def get_data_paths(directory: Optional[str] = None) -> DataPaths:
    """Ensure Himmelstein's data files are downloaded."""
    if directory is None:
        directory = DATA_DIRECTORY

    os.makedirs(directory, exist_ok=True)

    node_data_path = os.path.join(directory, 'nodes.tsv')
    if not os.path.exists(node_data_path):
        logger.info(f'downloading {NODE_DATA_URL}')
        urlretrieve(NODE_DATA_URL, node_data_path)

    edge_data_path = os.path.join(directory, 'edges.sif.gz')
    if not os.path.exists(edge_data_path):
        logger.info(f'downloading {EDGE_DATA_URL}')
        urlretrieve(EDGE_DATA_URL, edge_data_path)

    transformed_features_path = os.path.join(directory, 'transformed-features.tsv.bz2')
    if not os.path.exists(transformed_features_path):
        logger.info(f'downloading {TRANSFORMED_FEATURES_URL}')
        urlretrieve(TRANSFORMED_FEATURES_URL, transformed_features_path)

    validate_data_path = os.path.join(directory, 'validation-statuses.tsv')
    if not os.path.exists(validate_data_path):
        logger.info(f'downloading {VALIDATE_DATA_URL}')
        urlretrieve(VALIDATE_DATA_URL, validate_data_path)

    symptomatic_data_path = os.path.join(directory, 'probabilities.tsv')
    if not os.path.exists(symptomatic_data_path):
        logger.info(f'downloading {SYMPTOMATIC_DATA_URL}')
        urlretrieve(SYMPTOMATIC_DATA_URL, symptomatic_data_path)

    permutation_directory = os.path.join(directory, "permutations")
    os.makedirs(permutation_directory, exist_ok=True)

    permutation_paths = []
    for i in range(5):
        permutation_data_path = os.path.join(permutation_directory, PERMUTATION_DATA_FILE_FMT.format(i + 1))
        if not os.path.exists(permutation_data_path):
            url = PERMUTATION_DATA_URL_FMT.format(i + 1)
            logger.info(f'downloading {url}')
            urlretrieve(url, permutation_data_path)
        permutation_paths.append(permutation_data_path)
    data_edge2vec_path = os.path.join(directory, 'data_edge2vec')

    return DataPaths(
        node_data_path=node_data_path,
        edge_data_path=edge_data_path,
        transformed_features_path=transformed_features_path,
        validate_data_path=validate_data_path,
        symptomatic_data_path=symptomatic_data_path,
        permutation_paths=permutation_paths,
        data_edge2vec_path=data_edge2vec_path,
    )
