# -*- coding: utf-8 -*-

"""A class for loading previously trained models."""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import click
import joblib
import numpy as np
import pandas as pd
import rdkit
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .download import get_data_paths

__all__ = [
    'Predictor',
    'ConsensusPredictor',
]

logger = logging.getLogger(__name__)

Embeddings = Mapping[str, np.ndarray]
PredictionMapping = Mapping[str, float]
Aggregator = Callable[[np.ndarray], float]

data_paths = get_data_paths()
nodes_df = pd.read_csv(data_paths.node_data_path, sep='\t')
nodes_list = nodes_df['id'].tolist()
names_list = nodes_df['name'].tolist()
nodes_dict = dict(zip(nodes_list, names_list))
drugbank_url = 'https://raw.githubusercontent.com/dhimmel/drugbank/gh-pages/data/drugbank.tsv'


class BasePredictor:
    """Functions shared by all predictors."""

    def get_top_diseases(self, source_id: str, k: int = 30):
        """Get the top diseases for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> diseases = predictor.get_top_diseases('Compound::DB00997')

        Which diseases is HDAC6 involved in?
        >>> from drugrelink.default_predictor import predictor
        >>> diseases = predictor.get_top_diseases('Gene::10013')
        """
        return self._get_predictions_df(source_id, 'Disease::', k=k)

    def get_top_targets(self, source_id, k: int = 30):
        """Get the top targets for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> genes = predictor.get_top_targets('Compound::DB00997')
        """
        return self._get_predictions_df(source_id, 'Gene::', k=k)

    def get_top_chemicals(self, disease, k: int = 30):
        """Get the top chemicals for the given entity.

        >>> from drugrelink.default_predictor import predictor
        >>> chemicals = predictor.get_top_chemicals('Disease::DOID:1936')

        Which chemicals might inhibit HDAC6?
        >>> from drugrelink.default_predictor import predictor
        >>> chemicals = predictor.get_top_chemicals('Gene::10013')
        """
        return self._get_predictions_df(disease, 'Compound::', k=k)

    def _get_predictions_df(self, source_id, prefix, k: Optional[int] = None) -> pd.DataFrame:
        prediction_mapping: PredictionMapping = self._get_predictions_dict(source_id, prefix, k=k)

        rows = []
        for target, probability in prediction_mapping.items():
            target_type, target_id = target.split('::')
            rows.append((
                target_type,
                target_id,
                nodes_dict[target],
                probability,
                -np.log10(probability),
            ))

        df = pd.DataFrame(rows, columns=['type', 'id', 'name', 'p', 'mlp'])
        df.sort_values('mlp', ascending=False, inplace=True)
        return df

    def _get_predictions_dict(self, source_id, prefix, k: Optional[int] = None) -> PredictionMapping:
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
    """Make predictions using a single model."""

    #: Word2Vec model
    word2vec: Word2Vec

    #: Model trained on edge embeddings
    model: LogisticRegression

    #: Mappings from drugbank to chemical fingerprints
    _drugbank_id_to_maccs: Dict[str, Any] = field(default_factory=dict)

    @property
    def drugbank_to_maccs(self) -> Mapping[str, Any]:
        """The dictionary of drugbank identifiers to MACCSkeys from RDKit."""
        if self._drugbank_id_to_maccs:
            return self._drugbank_id_to_maccs

        logger.info('Loading DrugBank/InChI/MACCS key map')
        df = pd.read_csv(drugbank_url, sep='\t', names=['drugbank_id', 'inchi'])
        # Throw away all biotechs
        df = df[df.inchi.notna()]
        from rdkit.Chem import MACCSkeys, MolFromInchi

        for drugbank_id, inchi in tqdm(df):
            mol = MolFromInchi(inchi)
            if mol is None:
                logger.warning(f'Could not parse inchi: {inchi}')
            self._drugbank_id_to_maccs[drugbank_id] = MACCSkeys.GenMACCSKeys(mol)

        return self._drugbank_id_to_maccs

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

    def _get_predictions_dict(self, source_id, prefix, k: Optional[str] = None) -> PredictionMapping:
        target_ids = self._get_target_ids(source_id, prefix)
        source_embedding = self.word2vec.wv[source_id]
        return self._get_predictions_from_embedding(source_embedding, target_ids, k=k)

    def _get_predictions_from_embedding(
        self,
        source_embedding,
        target_ids,
        k: Optional[int] = None,
    ) -> PredictionMapping:
        edge_embeddings = np.asarray([
            source_embedding * self.word2vec.wv[target_id]  # uses hadamard
            for target_id in target_ids
        ])
        # We want to see low probabilities of class 0 (not an edge)
        # so p=0.001 means there's a 99.9% chance the edge should be there
        probabilities = self.model.predict_proba(edge_embeddings)[:, 0]

        if k is not None:
            return dict(zip(target_ids, probabilities))

        s = sorted(zip(target_ids, probabilities), key=itemgetter(1))
        return dict(s[:k])

    def _get_target_ids_by_prefix(self, prefix):
        return [
            target_id
            for target_id in self.word2vec.wv.index2word
            if target_id.startswith(prefix)
        ]

    def _get_target_ids(self, source_id, prefix):
        return [
            target_id
            for target_id in self._get_target_ids_by_prefix(prefix)
            if source_id != target_id
        ]

    def get_untrained_embedding(self, inchi: str, k: Optional[int] = None) -> PredictionMapping:
        similarities = self._get_chemical_similarities(inchi, k=k)
        embedding = self._get_untrained_embedding(similarities)
        target_ids = self._get_target_ids_by_prefix('Chemical::')
        return self._get_predictions_from_embedding(embedding, target_ids)

    def _get_chemical_similarities(self, inchi: str, k: Optional[int] = None) -> Mapping[str, float]:
        from rdkit import DataStructs
        from rdkit.Chem import MACCSkeys, MolFromInchi

        # 1. Parse
        mol = MolFromInchi(inchi)
        if mol is None:
            raise ValueError(f'could not parse InChI: {inchi}')

        # 2. Make fingerprint
        query_fingerprint = MACCSkeys.GenMACCSKeys(mol)

        rv = {
            name: DataStructs.FingerprintSimilarity(query_fingerprint, fingerprint)
            for name, fingerprint in self.drugbank_to_maccs.items()
        }

        if k is None:
            return rv

        sorted_items = sorted(
            rv.items(),
            key=itemgetter(1),
            reverse=True,
        )
        return dict(sorted_items[:k])

    def _get_untrained_embedding(self, similarities: Mapping[str, float]) -> np.ndarray:
        return np.sum([
            self.word2vec.wv[target_id] * similarity
            for target_id, similarity in similarities.items()
        ]) / sum(similarities.values())


@dataclass
class ConsensusPredictor(BasePredictor):
    """Make predictions using a consensus of several models."""

    predictors: List[Predictor]

    aggregator: Union[str, Aggregator] = 'min'

    def __post_init__(self):
        if isinstance(self.aggregator, str):
            if self.aggregator == 'min':
                self.aggregator = np.min
            elif self.aggregator == 'max':
                self.aggregator = np.max
            else:
                raise ValueError(f'invalid aggregator: {self.aggregator}')

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
            if os.path.isdir(os.path.join(directory, subdirectory))
        ]

        return cls(predictors)

    def _get_predictions_dict(self, source_id, prefix, k: Optional[int] = None) -> PredictionMapping:
        target_ids = self._get_target_ids(source_id, prefix)

        prediction_mappings: List[PredictionMapping] = [
            predictor._get_predictions_dict(source_id, prefix, k=k)
            for predictor in self.predictors
        ]

        return {
            target_id: self.aggregator([
                prediction_mapping[target_id]
                for prediction_mapping in prediction_mappings
            ])
            for target_id in target_ids
        }

    def _get_target_ids(self, source_id, prefix):
        return self.predictors[0]._get_target_ids(source_id, prefix)
