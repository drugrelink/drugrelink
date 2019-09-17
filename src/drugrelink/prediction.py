# -*- coding: utf-8 -*-

"""A class for loading previously trained models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

Embeddings = Mapping[str, np.ndarray]

__all__ = [
    'Predictor',
]


def load_embeddings(path: str) -> Embeddings:
    pass


@dataclass
class Predictor:
    """"""

    model: LogisticRegression
    embeddings: Embeddings

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str,
        embeddings_path: str,
    ) -> Predictor:
        """Generate a predictor."""
        model = joblib.load(model_path)
        embeddings = load_embeddings(embeddings_path)

        return Predictor(
            model=model,
            embeddings=embeddings,
        )

    def get_edge_probability(self, source_id: str, target_id: str) -> float:
        """Get the probability of the edge between the two nodes."""
        edge_embedding = self.get_edge_embedding(source_id, target_id)
        return self._predict_helper([edge_embedding.tolist()])[0]

    def get_edge_embedding(self, source_id: str, target_id: str) -> np.ndarray:
        """Get the embedding of the edge between the two nodes."""
        return self.embeddings[source_id] * self.embeddings[target_id]

    def _predict_helper(self, edge_embedding):
        # FIXME is this a 0 or a 1?!?!
        return self.model.predict_proba(edge_embedding)[:, 0]

    def get_top_diseases(self, drug: str, k: int = 30):
        """Get the top diseases for the given drug.

        >>> from drugrelink.default_predictor import predictor
        >>> predictor.get_top_diseases('Compound::DB00997')
        ...
        """

    def get_top_chemicals(self, disease, k: int = 30):
        """Get the top chemicals for the given disease.

        >>> from drugrelink.default_predictor import predictor
        >>> predictor.get_top_diseases('Disease::DOID:1936')
        ...
        """
