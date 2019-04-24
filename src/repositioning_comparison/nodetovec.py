""""""

from typing import Iterable, Tuple, List, Type

import gensim
import networkx
import node2vec


def nodetovec(graph: networkx.Graph) -> gensim.models.Word2Vec:
    n_model = node2vec.Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = n_model.fit(window=10, min_count=1, batch_words=4)
    return model


def _get_edge_vectors(
        model: node2vec.Node2Vec,
        pair_list: Iterable[Tuple[str, str]],
        embedder: Type[node2vec.edges.EdgeEmbedder],
) -> List:
    """

    :param model:
    :param pair_list:
    :return:
    """
    edges_embs = embedder(keyed_vectors=model.wv)
    return [
        edges_embs[(node1, node2)].tolist()
        for node1, node2 in pair_list
    ]


def HadamardEmbedder(model: node2vec.Node2Vec, pair_list: Iterable[Tuple]):
    """

    :param model:
    :param pair_list:
    :return:
    """
    return _get_edge_vectors(model, pair_list, node2vec.edges.HadamardEmbedder)


def AverageEmbedder(model: node2vec.Node2Vec, pair_list):
    return _get_edge_vectors(model, pair_list, node2vec.edges.AverageEmbedder)


def WeightedL1Embedder(model: node2vec.Node2Vec, pair_list):
    return _get_edge_vectors(model, pair_list, node2vec.edges.WeightedL1Embedder)


def WeightedL2Embedder(model: node2vec.Node2Vec, pair_list):
    return _get_edge_vectors(model, pair_list, node2vec.edges.WeightedL2Embedder)
