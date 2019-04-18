import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from pathlib import Path

def graph(nodepath,edgepath):
    #import nodes and edges from data as df

    dfhimnode = pd.read_csv(nodepath,sep='\t')
    dfhimedge = pd.read_csv(edgepath,sep='\t')
    
    # create graph
    himgraph = nx.Graph()
    for index,row in dfhimnode.iterrows():
        if row['id'] not in himgraph.nodes():
            himgraph.add_node(row['id'],name= row['name'],kind=row['kind'])
    for index,row in dfhimedge.iterrows():
        himgraph.add_edge(row['source'],row['target'],metaedge = row['metaedge'])
    return himgraph
