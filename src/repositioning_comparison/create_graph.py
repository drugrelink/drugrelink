# -*- coding: utf-8 -*-

import gzip
import logging

import networkx as nx
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_himmelstein_graph(node_path: str, edge_path: str) -> nx.Graph:
    """Import nodes and edges from data as df."""
    logger.info('reading edges')
    with gzip.open(edge_path, 'rb') as file:
        edge_df = pd.read_csv(file, sep='\t')  # TODO only read relevant columns

    graph = nx.from_pandas_edgelist(edge_df, edge_attr='metaedge')

    logger.info('reading nodes')  # TODO only read relevant columns
    node_df = pd.read_csv(node_path, sep='\t')
    for _, row in tqdm(node_df.iterrows(), total=len(node_df.index), desc='adding nodes'):
        if row['id'] not in graph:
            graph.add_node(row['id'], name=row['name'], kind=row['kind'])

    return graph
