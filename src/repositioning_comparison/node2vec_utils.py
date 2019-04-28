# -*- coding: utf-8 -*-

import gensim
import networkx
import node2vec


def fit_node2vec(graph: networkx.Graph, workers: int = 4) -> gensim.models.Word2Vec:
    node2vec_model = node2vec.Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=workers)
    word2vec_model = node2vec_model.fit(window=10, min_count=1, batch_words=4)
    return word2vec_model
