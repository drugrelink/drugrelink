# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
from glmnet.logistic import LogitNet
from sklearn.metrics import roc_auc_score, average_precision_score


def train_logistic_regression(x, y) -> LogitNet:
    logit_net = LogitNet(
        alpha=0.2,
        n_lambda=150,
        min_lambda_ratio=1e-8,
        n_jobs=10,
        random_state=2,
    )
    logit_net.fit(x, y)
    return logit_net


def validate(logit_net: LogitNet, x, y) :
    x = np.array(x)
    y_pred_probab = logit_net.predict_proba(x)
    y_pred_labels = logit_net.predict(x)
    predicted_edge_probabilities = [
        score[1]
        for score in y_pred_probab
    ]
    roc = roc_auc_score(y, predicted_edge_probabilities)
    aupr = average_precision_score(y,predicted_edge_probabilities)
    return roc, y_pred_probab, y_pred_labels, aupr
