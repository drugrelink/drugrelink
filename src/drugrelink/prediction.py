# -*- coding: utf-8 -*-

"""A class for loading previously trained models."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Mapping
import pandas as pd
from glmnet.logistic import LogitNet

import joblib
import numpy as np
from gensim.models import Word2Vec
from node2vec.edges import EdgeEmbedder, HadamardEmbedder
from sklearn.linear_model import LogisticRegression
from .download import get_data_paths



data_paths = get_data_paths()
node_path = data_paths.node_data_path
nodes_df = pd.read_csv (node_path,sep='\t')
nodes_list = nodes_df['id'].tolist()
names_list = nodes_df ['name'].tolist()
nodes_dict = dict(zip(nodes_list,names_list))
Embeddings = Mapping[str, np.ndarray]
__all__ = [
    'Predictor',
]


@dataclass
class Predictor:
    """"""

    #: Word2Vec model
    word2vec: Word2Vec

    #: Model trained on edge embeddings
    model: LogisticRegression

    #: The edge embedder
    edge_embedder: EdgeEmbedder = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived fields."""
        self.edge_embedder = HadamardEmbedder(self.word2vec.wv)
    @classmethod
    def from_paths(
        model_path: str,
        word2vec_path: str,
    ) -> Predictor:
        """Generate a predictor."""

        model = joblib.load(model_path)

        with open(word2vec_path, 'rb') as file:
            word2vec = pickle.load(file)

        return Predictor(
            model=model,
            word2vec=word2vec,
        )

    def get_top_diseases(self, source_id: str, k: int = 30):
        """Get the top diseases for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> predictor.get_top_diseases('Compound::DB00997')
        ...

        Which diseases is HDAC6 involved in?
        >>> from drugrelink.default_predictor import predictor
        >>> predictor.get_top_diseases('Gene::10013')
        """
        return self._get_top_prefixed(source_id, 'Disease::')

    def get_top_targets(self, source_id, k: int = 30):
        """Get the top targets for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> predictor.get_top_diseases('Compound::DB00997')
        ...
        """
        return self._get_top_prefixed(source_id, 'Disease::')

    def get_top_chemicals(self, disease, k: int = 30):
        """Get the top chemicals for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> predictor.get_top_diseases('Disease::DOID:1936')
        ...

        Which chemicals might inhibit HDAC6?
        >>> from drugrelink.default_predictor import predictor
        >>> predictor.get_top_chemicals('Gene::10013')
        """
        return self._get_top_prefixed(disease, 'Compound::')

    def _get_top_prefixed(self, source_id, prefix) -> Mapping[str, np.ndarray]:
        target_ids = [
            target_id
            for target_id in self.word2vec.wv.index2word
            if source_id != target_id and target_id.startswith(prefix)
        ]
        edge_embeddings =np.asarray( [
            self.edge_embedder[source_id, target_id]
            for target_id in target_ids

        ])
        target_names = [nodes_dict[target_id]
                        for target_id in target_ids]
        probabilities = self.model.predict_proba(edge_embeddings)[:,0]
        print(dict(zip(target_names, probabilities)))
        return target_names,probabilities
