"""Helper functions for getting resources."""

import logging
import os
from typing import Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

HERE = os.path.abspath(os.path.dirname(__file__))

DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'data'))
DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)

NODE_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/nodes.tsv'
EDGE_DATA_URL = 'https://raw.githubusercontent.com/dhimmel/integrate/master/data/edges.sif.gz'
TRANSFORMED_FEATURES_URL = 'https://github.com/dhimmel/learn/blob/master/prediction/features/transformed-features.tsv.bz2?raw=true'


def ensure_data(directory: Optional[str] = None) -> Tuple[str, str, str]:
    """Ensure Himmelstein's data files are downloaded."""
    if directory is None:
        directory = DIRECTORY

    os.makedirs(directory, exist_ok=True)

    NODE_DATA_PATH = os.path.join(directory, 'nodes.tsv')
    if not os.path.exists(NODE_DATA_PATH):
        logger.warning(f'downloading {NODE_DATA_URL}')
        urlretrieve(NODE_DATA_URL, NODE_DATA_PATH)

    EDGE_DATA_PATH = os.path.join(directory, 'edges.sif.gz')
    if not os.path.exists(EDGE_DATA_PATH):
        logger.warning(f'downloading {EDGE_DATA_URL}')
        urlretrieve(EDGE_DATA_URL, EDGE_DATA_PATH)

    TRANSFORMED_FEATURES_PATH = os.path.join(directory, 'transformed-features.tsv.bz2')
    if not os.path.exists(TRANSFORMED_FEATURES_PATH):
        logger.warning(f'downloading {TRANSFORMED_FEATURES_URL}')
        urlretrieve(TRANSFORMED_FEATURES_URL, TRANSFORMED_FEATURES_PATH)

    return NODE_DATA_PATH, EDGE_DATA_PATH, TRANSFORMED_FEATURES_PATH
