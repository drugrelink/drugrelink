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
    hetio.readwrite.write_nodetable(him_per, nodepath)
    hetio.readwrite.write_sif(him_per, edgepath)
    dfhimnode = pd.read_csv(nodepath, sep='\t')
    dfhimedge = pd.read_csv(edgepath, sep='\t')

    himgraph = nx.Graph()
    for index, row in dfhimnode.iterrows():
        if row['id'] not in himgraph.nodes():
            himgraph.add_node(row['id'], name=row['name'], kind=row['kind'])
    for index, row in dfhimedge.iterrows():
        himgraph.add_edge(row['source'], row['target'], metaedge=row['metaedge'])
    return himgraph
