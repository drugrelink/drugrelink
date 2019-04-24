from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train(x, y):
    lg = LogisticRegression(random_state=0, solver='lbfgs', tol=1e-8).fit(x, y)
    return lg


def validate(lg, x, y):
    roc = roc_auc_score(y, lg.predict_proba(x)[:, 1])
    return roc
