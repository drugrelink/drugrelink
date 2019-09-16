# -*- coding: utf-8 -*-

"""Default predictor for :mod:`drugrelink`."""

import os

from .constants import RESOURCES_DIRECTORY
from .prediction import Predictor

__all__ = [
    'predictor',
]

PREDICTIVE_MODEL_DIRECTORY = os.path.join(RESOURCES_DIRECTORY, 'predictive_model', '0')

DEFAULT_MODEL_PATH = os.path.join(PREDICTIVE_MODEL_DIRECTORY, 'logistic_regression_clf.joblib')
DEFAULT_EMBEDDINGS_PATH = os.path.join(PREDICTIVE_MODEL_DIRECTORY, 'word2vec_wv')

predictor = Predictor.from_paths(
    model_path=DEFAULT_MODEL_PATH,
    embeddings_path=DEFAULT_EMBEDDINGS_PATH,
)
