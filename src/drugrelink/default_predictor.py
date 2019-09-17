# -*- coding: utf-8 -*-

"""Default predictor for :mod:`drugrelink`."""

import json
import os

import click

from .constants import RESOURCES_DIRECTORY
from .prediction import Predictor

__all__ = [
    'predictor',
]

EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY = os.path.join(RESOURCES_DIRECTORY, 'predictive_model', 'edge2vec', '0')
assert os.path.exists(EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY) and os.path.isdir(EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY)

DEFAULT_MODEL_PATH = os.path.join(EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY, 'logistic_regression_clf.joblib')
assert os.path.exists(DEFAULT_MODEL_PATH)

DEFAULT_WORD2VEC_MODEL_PATH = os.path.join(EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY, 'word2vec_model.pickle')
assert os.path.exists(DEFAULT_WORD2VEC_MODEL_PATH)

predictor = Predictor.from_paths(
    model_path=DEFAULT_MODEL_PATH,
    word2vec_path=DEFAULT_WORD2VEC_MODEL_PATH,
)


@click.command()
@click.option('-c', '--chemical-id', type=str)
@click.option('-d', '--disease-id', type=str)
@click.option('-m', '--method', typte=click.Choice(['node2vec', 'edge2vec']), default='node2vec')
def main(chemical_id, disease_id, method):
    """Predict diseases for the given chemical.

    Use ``drugrelink-repurpose DB00997`` to show examples
    for `Doxorubicin <https://identifies.org/drugbank:DB00997>`_.
    """
    if method == 'node2vec':
        pass
    elif method == 'edge2vec':
        if chemical_id:
            if chemical_id.startswith('drugbank:'):
                chemical_id = chemical_id[len('drugbank:'):]
            if not chemical_id.startswith('Compound::'):
                chemical_id = f'Compound::{chemical_id}'
            predictions = predictor.get_top_diseases(chemical_id)
            click.echo(json.dumps(predictions, indent=2))

        if disease_id:
            if disease_id.startswith('DOID:'):
                disease_id = f'Disease::{disease_id}'
            if not disease_id.startswith('Disease::'):
                disease_id = f'Disease::{disease_id}'
            predictions = predictor.get_top_chemicals(disease_id)
            click.echo(json.dumps(predictions, indent=2))


if __name__ == '__main__':
    main()
