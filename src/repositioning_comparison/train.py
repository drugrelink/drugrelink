# -*- coding: utf-8 -*-
import pandas as pd
from glmnet import logistic
from sklearn.metrics import roc_auc_score


def train_logistic_regression(x, y) -> logistic.LogitNet:
    logistic_regression = lg = logistic.LogitNet(alpha=0.2,n_lambda=150,min_lambda_ratio=1e-8, n_jobs=10,random_state=2)
    logistic_regression.fit(x, y)
    return logistic_regression


def validate(logistic_regression: logistic.LogitNet, x, y) -> float:
    print(logistic_regression.predict_proba(x))
    roc = roc_auc_score(y, logistic_regression.predict_proba(x))
    y_pro = logistic_regression.predict_proba(x)
    return roc, y_pro
