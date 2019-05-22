# -*- coding: utf-8 -*-

"""A wrapper around Node2vec."""

import logging
import multiprocessing
import os
import pickle
from typing import Optional

import gensim
import networkx
from node2vec import Node2Vec
from node2vec.parallel import parallel_generate_walks

logger = logging.getLogger(__name__)


class _SubNode2Vec(Node2Vec):
    """A subclass of Node2Vec that gives a bit more logging."""

    def __init__(self, *args, transition_probabilities_path: Optional[str] = None, **kwargs) -> None:
        self.transition_probabilities_path = transition_probabilities_path
        super().__init__(*args, **kwargs)

    def _precompute_probabilities(self):
        if self.transition_probabilities_path is not None and os.path.exists(self.transition_probabilities_path):
            with open(self.transition_probabilities_path, 'rb') as file:
                self.d_graph = pickle.load(file)
                logger.warning(f'Loaded pre-computed probabilities from {self.transition_probabilities_path}')
                return

        logger.warning('Begin pre-computing probabilities')
        super()._precompute_probabilities()
        logger.warning('Finished pre-computing probabilities')

        if self.transition_probabilities_path is not None:
            logger.warning(f'Dumping pre-computed probabilities to {self.transition_probabilities_path}')
            with open(self.transition_probabilities_path, 'wb') as file:
                pickle.dump(self.d_graph, file)

    def _generate_walks(self):
        """Generate the random walks which will be used as the skip-gram input.

        :return: List of walks. Each walk is a list of nodes.
        """
        return parallel_generate_walks(
            self.d_graph,
            self.walk_length,
            self.num_walks,
            'Single process!',
            self.sampling_strategy,
            self.NUM_WALKS_KEY,
            self.WALK_LENGTH_KEY,
            self.NEIGHBORS_KEY,
            self.PROBABILITIES_KEY,
            self.FIRST_TRAVEL_KEY,
            self.quiet
        )


def fit_node2vec(
        graph: networkx.Graph,
        transition_probabilities_path: Optional[str] = None,
        workers: int = 1,
        walk_length=30,
        dimensions=64,
        num_walks=10,
        window=10

) -> gensim.models.Word2Vec:
    """

    :param graph:
    :param transition_probabilities_path: Place to dump pre-computed transition probability matrix
    :param workers: The number of workers to use. If -1, uses all available CPUs
    """
    if workers == 0:
        raise ValueError(f'Invalid number of workers: {workers}')
    elif workers == -1:
        workers = multiprocessing.cpu_count()
    elif workers < -1:
        workers = multiprocessing.cpu_count() + workers + 1

    logger.info(f"Initializing Node2Vec with {workers}")

    node2vec_model = _SubNode2Vec(
        graph=graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        transition_probabilities_path=transition_probabilities_path,
    )
    word2vec_model = node2vec_model.fit(window=window, min_count=1, batch_words=4)
    return word2vec_model
