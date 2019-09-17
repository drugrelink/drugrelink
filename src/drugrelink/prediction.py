# -*- coding: utf-8 -*-

"""A class for loading previously trained models."""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, List, Mapping

import click
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from node2vec.edges import EdgeEmbedder, HadamardEmbedder
from sklearn.linear_model import LogisticRegression

from .download import get_data_paths

__all__ = [
    'Predictor',
    'ConsensusPredictor',
]

logger = logging.getLogger(__name__)

Embeddings = Mapping[str, np.ndarray]

data_paths = get_data_paths()
nodes_df = pd.read_csv(data_paths.node_data_path, sep='\t')
nodes_list = nodes_df['id'].tolist()
names_list = nodes_df['name'].tolist()
nodes_dict = dict(zip(nodes_list, names_list))


class BasePredictor:

    def get_top_diseases(self, source_id: str, k: int = 30):
        """Get the top diseases for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> diseases = predictor.get_top_diseases('Compound::DB00997')

        Which diseases is HDAC6 involved in?
        >>> from drugrelink.default_predictor import predictor
        >>> diseases = predictor.get_top_diseases('Gene::10013')
        """
        return self._get_predictions_df(source_id, 'Disease::')

    def get_top_targets(self, source_id, k: int = 30):
        """Get the top targets for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> genes = predictor.get_top_targets('Compound::DB00997')
        """
        return self._get_predictions_df(source_id, 'Gene::')

    def get_top_chemicals(self, disease, k: int = 30):
        """Get the top chemicals for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> chemicals = predictor.get_top_chemicals('Disease::DOID:1936')

        Which chemicals might inhibit HDAC6?
        >>> from drugrelink.default_predictor import predictor
        >>> chemicals = predictor.get_top_chemicals('Gene::10013')
        """
        return self._get_predictions_df(disease, 'Compound::')

    def _get_predictions_df(self, source_id, prefix) -> pd.DataFrame:
        d = self._get_predictions_dict(source_id, prefix)

        rows = []
        for target, probability in d.items():
            target_type, target_id = target.split('::')
            rows.append((
                target_type,
                target_id,
                nodes_dict[target],
                probability,
                -np.log10(probability),
            ))

        df = pd.DataFrame(rows, columns=['type', 'id', 'name', 'p', 'mlp']).sort_values('mlp', ascending=False)
        return df

    def _get_predictions_dict(self, source_id, prefix) -> Mapping[str, float]:
        raise NotImplementedError

    def get_cli(self) -> click.Command:
        """Get a :mod:`click` command."""

        @click.command()
        @click.argument('chemical-id')
        def main(chemical_id):
            """Predict diseases for the given chemical.

            Use ``drugrelink-repurpose DB00997`` to show examples
            for `Doxorubicin <https://identifies.org/drugbank:DB00997>`_.
            """
            logging.basicConfig(level=logging.INFO)
            logger.setLevel(logging.INFO)

            if chemical_id.startswith('drugbank:'):
                chemical_id = chemical_id[len('drugbank:'):]

            if not chemical_id.startswith('Compound::'):
                chemical_id = f'Compound::{chemical_id}'

            predictions = self.get_top_diseases(chemical_id)
            print(predictions)
            # click.echo(json.dumps(predictions, indent=2))

        return main


@dataclass
class Predictor(BasePredictor):
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
        cls,
        *,
        word2vec_path: str,
        model_path: str,
    ) -> Predictor:
        """Generate a predictor."""
        logger.info(f'Loading Word2Vec model from {word2vec_path}')
        with open(word2vec_path, 'rb') as file:
            word2vec = pickle.load(file)

        logger.info(f'Loading LogitNet model from {model_path}')
        model = joblib.load(model_path)

        return Predictor(
            word2vec=word2vec,
            model=model,
        )

    def _get_predictions_dict(self, source_id, prefix) -> Mapping[str, float]:
        target_ids = self._get_target_ids(source_id, prefix)
        edge_embeddings = np.asarray([
            self.edge_embedder[source_id, target_id]
            for target_id in target_ids
        ])
        probabilities = self.model.predict_proba(edge_embeddings)[:, 0]
        return dict(zip(target_ids, probabilities))

    def _get_target_ids(self, source_id, prefix):
        return [
            target_id
            for target_id in self.word2vec.wv.index2word
            if source_id != target_id and target_id.startswith(prefix)
        ]


@dataclass
class ConsensusPredictor(BasePredictor):
    """Make predictions using a consensus of several models."""

    predictors: List[Predictor]

    aggregator: Callable[[np.ndarray], float] = np.min

    @classmethod
    def from_directory(
        cls,
        directory,
        lr_name: str = 'logistic_regression_clf.joblib',
        word2vec_name: str = 'word2vec_model.pickle'
    ) -> ConsensusPredictor:
        """Make a consensus predictor using the models in subdirectories of the given directory."""
        assert os.path.exists(directory)
        assert os.path.isdir(directory)

        predictors = [
            Predictor.from_paths(
                model_path=os.path.join(directory, subdirectory, lr_name),
                word2vec_path=os.path.join(directory, subdirectory, word2vec_name)
            )
            for subdirectory in os.listdir(directory)
        ]

        return cls(predictors)

    def _get_predictions_dict(self, source_id, prefix) -> Mapping[str, float]:
        target_ids = self._get_target_ids(source_id, prefix)

        ds = [
            predictor._get_predictions_dict(source_id, prefix)
            for predictor in self.predictors
        ]

        return {
            target_id: self.aggregator([d[target_id] for d in ds])
            for target_id in target_ids
        }

    def _get_target_ids(self, source_id, prefix):
        return self.predictors[0]._get_target_ids(source_id, prefix)
