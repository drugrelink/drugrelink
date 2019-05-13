# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_logistic_regression(x, y) -> LogisticRegression:
    logistic_regression = LogisticRegression(random_state=0, solver='lbfgs', tol=1e-8)
    logistic_regression.fit(x, y)
    return logistic_regression


def validate(logistic_regression: LogisticRegression, x, y) -> float:
    roc = roc_auc_score(y, logistic_regression.predict_proba(x)[:, 1])
    y_pro = 
    return roc
