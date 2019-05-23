# -*- coding: utf-8 -*-

import os

import hetio
import networkx as nx
import pandas as pd
from hetio import readwrite

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'data', 'permutations'))
PERMUTATION_DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)


def convert(path,order):
    him_per = readwrite.read_graph(path)
    nodepath = os.path.join(PERMUTATION_DIRECTORY,'permutation_node{}.tsv'.format(order))
    edgepath = os.path.join(PERMUTATION_DIRECTORY,'permutation_edge{}.sif.gz'.format(order))
    with open(nodepath,'w') as node_path:
        hetio.readwrite.write_nodetable(him_per, node_path)
    with open(edgepath,'w') as edge_path:
        hetio.readwrite.write_sif(him_per, edge_path)
    dfhimnode = pd.read_csv(node_path, sep='\t')

    dfhimedge = pd.read_csv(edge_path, sep='\t')

    himgraph = nx.Graph()
    for index, row in dfhimnode.iterrows():
        if row['id'] not in himgraph.nodes():
            himgraph.add_node(row['id'], name=row['name'], kind=row['kind'])
    for index, row in dfhimedge.iterrows():
        himgraph.add_edge(row['source'], row['target'], metaedge=row['metaedge'])
    return himgraph
