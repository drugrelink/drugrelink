import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from pathlib import Path

def graph():
    #import nodes and edges from data as df
    path_node = str(Path().absolute())
    dfhimnode = pd.read_csv(path_node+'/original_graph_data/nodes.tsv',sep='\t')
    path_edge = str(Path().absolute())
    dfhimedge = pd.read_csv(path_edge+'/original_graph_data/edges.sif',sep='\t')
    
    # create graph
    himgraph = nx.Graph()
    for index,row in dfhimnode.iterrows():
        if row['id'] not in himgraph.nodes():
            himgraph.add_node(row['id'],name= row['name'],kind=row['kind'])
    for index,row in dfhimedge.iterrows():
        himgraph.add_edge(row['source'],row['target'],metaedge = row['metaedge'])
    return himgraph
