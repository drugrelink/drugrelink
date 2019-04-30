# -*- coding: utf-8 -*-

import logging
import multiprocessing

import gensim
import networkx
import node2vec

logger = logging.getLogger(__name__)


def fit_node2vec(graph: networkx.Graph, workers: int = -1) -> gensim.models.Word2Vec:
    """

    :param graph:
    :param workers: The number of workers to use. If -1, uses all available CPUs
    """
    if workers == 0:
        raise ValueError(f'Invalid number of workers: {workers}')
    elif workers == -1:
        workers = multiprocessing.cpu_count()
    elif workers < -1:
        workers = multiprocessing.cpu_count() + workers + 1

    logger.info(f"Initializing Node2Vec with {workers}")

    node2vec_model = node2vec.Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=workers)
    word2vec_model = node2vec_model.fit(window=10, min_count=1, batch_words=4)
    return word2vec_model
