# -*- coding: utf-8 -*-

import bz2

import networkx as nx
import pandas as pd


def generate_subgraph(path, graph, *, cutoff: int, positive_number: int, negative_number: int):
    """

    :param path:
    :param graph:
    :param cutoff:
    :param positive_number:
    :param negative_number:
    :return:
    """
    with bz2.BZ2File(path, 'r') as file:
        df_features = pd.read_csv(file, sep='\t', low_memory=False)

    df_positive = df_features.loc[df_features['status'] == 1][:positive_number]
    df_negative = df_features.loc[df_features['status'] == 0][:negative_number]
    subgraph_nodeslist = []

    positive_list = []
    positive_label = [1] * positive_number
    negative_list = []
    negative_label = [0] * negative_number

    for _, row in df_positive.iterrows():
        source = 'Compound::' + row['compound_id']
        target = 'Disease::' + row['disease_id']
        positive_list.append([source, target])
        for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=cutoff):
            subgraph_nodeslist = subgraph_nodeslist + path

    for _, row in df_negative.iterrows():
        source = 'Compound::' + row['compound_id']
        target = 'Disease::' + row['disease_id']
        negative_list.append([source, target])
        for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=cutoff):
            subgraph_nodeslist = subgraph_nodeslist + path

    subgraph_nodes_list = list(set(subgraph_nodeslist))
    subgraph = graph.subgraph(subgraph_nodes_list)

    return subgraph, positive_list, positive_label, negative_list, negative_label
