# -*- coding: utf-8 -*-

"""Functions for generating sub-graphs appropriate for learning."""

import bz2

import networkx as nx
import pandas as pd


def generate_subgraph(
        path,
        graph,
        *,
        n_positive: int,
        n_negative: int,
        max_simple_path_length: int = 3,
):
    """

    :param path:
    :param graph:
    :param n_positive:
    :param n_negative:
    :param max_simple_path_length: Maximum simple path length
    """
    with bz2.BZ2File(path, 'r') as file:
        df_features = pd.read_csv(file, sep='\t', low_memory=False)

    subgraph_nodes = set()

    positive_list = []
    positive_labels = [1] * n_positive
    df_positive = df_features.loc[df_features['status'] == 1][:n_positive]
    for _, row in df_positive.iterrows():
        source = 'Compound::' + row['compound_id']
        target = 'Disease::' + row['disease_id']
        positive_list.append([source, target])
        for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=max_simple_path_length):
            subgraph_nodes.update(path)

    negative_list = []
    negative_labels = [0] * n_negative
    df_negative = df_features.loc[df_features['status'] == 0][:n_negative]
    for _, row in df_negative.iterrows():
        source = 'Compound::' + row['compound_id']
        target = 'Disease::' + row['disease_id']
        negative_list.append([source, target])
        for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=max_simple_path_length):
            subgraph_nodes.update(path)

    subgraph = graph.subgraph(subgraph_nodes)

    return subgraph, positive_list, positive_labels, negative_list, negative_labels
