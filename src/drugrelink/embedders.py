# -*- coding: utf-8 -*-

"""Embedders for node2vec."""

from functools import partial
from typing import Iterable, List, Mapping, Optional, Tuple, Type

import gensim
from node2vec.edges import AverageEmbedder, EdgeEmbedder, HadamardEmbedder, WeightedL1Embedder, WeightedL2Embedder

from .typing import EmbedderFunction

__all__ = [
    'hadamard',
    'average',
    'weighted_l1',
    'weighted_l2',
    'EMBEDDERS',
    'get_embedder',
]


def embed(
    word2vec_model: gensim.models.Word2Vec,
    edges: Iterable[Tuple[str, str]],
    edge_embedder_cls: Type[EdgeEmbedder],
) -> List[List[float]]:
    """

    :param word2vec_model:
    :param edges: Pairs of source and target nodes
    :param edge_embedder_cls: Which :class:`node2vec.edges.EdgeEmbedder` from :mod:`node2vec.edges` to use.
    :return:
    """
    edge_embeddings = edge_embedder_cls(keyed_vectors=word2vec_model.wv)
    return [
        edge_embeddings[source, target].tolist()
        for source, target in edges
    ]


hadamard = partial(embed, edge_embedder_cls=HadamardEmbedder)
average = partial(embed, edge_embedder_cls=AverageEmbedder)
weighted_l1 = partial(embed, edge_embedder_cls=WeightedL1Embedder)
weighted_l2 = partial(embed, edge_embedder_cls=WeightedL2Embedder)

EMBEDDERS: Mapping[str, EmbedderFunction] = {
    'hadamard': hadamard,
    'average': average,
    'weighted_l1': weighted_l1,
    'weighted_l2': weighted_l2,
}


def get_embedder(name: Optional[str] = None) -> EmbedderFunction:
    """Get the embedder."""
    return EMBEDDERS.get(name, hadamard)
