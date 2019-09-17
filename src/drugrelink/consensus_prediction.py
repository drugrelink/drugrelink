# -*- coding: utf-8 -*-
""""""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Mapping
import pandas as pd
from glmnet.logistic import LogitNet
import os
import click
import json

import joblib
import numpy as np
from gensim.models import Word2Vec
from node2vec.edges import EdgeEmbedder, HadamardEmbedder
from sklearn.linear_model import LogisticRegression
from .download import get_data_paths
from .constants import RESOURCES_DIRECTORY
from .prediction import Predictor
__all__ = [
    'ConsensusPrediction',
]

@dataclass
class ConsensusPrediction:
    """"""

    method:str
    chemical:str = None
    disease: str = None
    output_directory:str=None
    def __init__(self,output_directory,method,chemical,disease):

        self.n_train =10
        self.chemical = chemical
        self.method = method
    def consensus(self):
        all_names = []
        all_p = []
        for i in range(self.n_train):
            lg_models_directory = os.path.join(RESOURCES_DIRECTORY,'predictive_model',self.method)
            model_path = os.path.join(lg_models_directory,str(i),'logistic_regression_clf.joblib')
            word2vec_path = os.path.join(lg_models_directory,str(i),'word2vec_model.pickle')
            predictor = Predictor.from_paths(model_path, word2vec_path)
            if self.chemical:
                target_names, probabilities = predictor.get_top_diseases(self.chemical)
            elif self.disease:
                target_names, probabilities = predictor.get_top_chemicals(self.disease)
            all_names =all_names + target_names
            all_p = all_p + probabilities
        dict_all = dict(zip(all_names,all_p))
        click.echo(json.dumps(dict_all, indent=2))
        return dict_all
