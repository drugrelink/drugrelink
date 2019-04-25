import tarfile

import networkx as nx
import pandas as pd
import gzip


def create_himmelstein_graph(nodepath, edgepath):
    # import nodes and edges from data as df

    dfhimnode = pd.read_csv(nodepath, sep='\t')
    f = gzip.open(edgepath,'rb')
    dfhimedge = pd.read_csv(f,sep='\t')

    # create graph
    himgraph = nx.Graph()
    for index, row in dfhimnode.iterrows():
        if row['id'] not in himgraph.nodes():
            himgraph.add_node(row['id'], name=row['name'], kind=row['kind'])
    for index, row in dfhimedge.iterrows():
        himgraph.add_edge(row['source'], row['target'], metaedge=row['metaedge'])

    return himgraph
