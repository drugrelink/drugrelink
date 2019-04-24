import tarfile

import networkx as nx
import pandas as pd


def subgraph(path, graph, cutoff, pnumber, nnumber):
    tar = tarfile.open(path, "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        df_features = pd.read_csv(f, sep='\t', low_memory=False)
    df_positive = df_features.loc[df_features['status'] == 1][0:pnumber]
    df_negative = df_features.loc[df_features['status'] == 0][0:nnumber]
    subgraph_nodeslist = []
    positive_list = []
    positive_label = [1] * pnumber
    negative_list = []
    negative_label = [0] * nnumber
    for index, row in df_positive.iterrows():
        positive_list.append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
        for path in nx.all_simple_paths(graph, source='Compound::' + row['compound_id'],
                                        target='Disease::' + row['disease_id'], cutoff=cutoff):
            subgraph_nodeslist = subgraph_nodeslist + path
    for i, r in df_negative.iterrows():
        negative_list.append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
        for path in nx.all_simple_paths(graph, source='Compound::' + r['compound_id'],
                                        target='Disease::' + r['disease_id'], cutoff=cutoff):
            subgraph_nodeslist = subgraph_nodeslist + path
    subgraph_nodes_list = list(set(subgraph_nodeslist))
    sub_graph = graph.subgraph(subgraph_nodes_list)

    return sub_graph, positive_list, positive_label, negative_list, negative_label
