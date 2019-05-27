# -*- coding: utf-8 -*-
import numpy as np
from glmnet import logistic
from sklearn.metrics import roc_auc_score


def train_logistic_regression(x, y) -> logistic.LogitNet:
    logistic_regression = lg = logistic.LogitNet(alpha=0.2, n_lambda=150, min_lambda_ratio=1e-8, n_jobs=10,
                                                 random_state=2)
    logistic_regression.fit(x, y)
    return logistic_regression


def validate(logistic_regression: logistic.LogitNet, x, y) -> float:
    x = np.array(x)
    scores = logistic_regression.predict_proba(x)
    label = logistic_regression.predict(x)
    y_score = []
    for i in scores:
        y_score.append(i[1])
    roc = roc_auc_score(y, y_score)

    return roc, scores,label
