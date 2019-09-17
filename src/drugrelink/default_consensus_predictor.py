# -*- coding: utf-8 -*-

"""Default consensus predictor for :mod:`drugrelink`.

Run on Doxorubicin with ``python -m drugrelink.default_consensus_predictor DB00997``.
"""

import os

from .constants import RESOURCES_DIRECTORY
from .prediction import ConsensusPredictor

__all__ = [
    'predictor',
]

EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY = os.path.join(RESOURCES_DIRECTORY, 'predictive_model', 'edge2vec')
assert os.path.exists(EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY) and os.path.isdir(EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY)

predictor = ConsensusPredictor.from_directory(EDGE2VEC_PREDICTIVE_MODEL_DIRECTORY)
consensus_predict = predictor.get_cli()

if __name__ == '__main__':
    consensus_predict()
