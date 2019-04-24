import tarfile

import pandas as pd


def train_pairs(path):
    tar = tarfile.open(path, "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        df_features = pd.read_csv(f, sep='\t', low_memory=False)

    train_list = []
    train_label = []
    for _, row in df_features[['compound_id', 'disease_id', 'status']].iterrows():
        train_list.append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
        train_label.append(row['status'])
    return train_list, train_label


def test_pairs(path):
    df_features = pd.read_csv(path, sep='\t')
    test_list = []
    test_label = []
    for _, row in df_features[
        ['compound_id', 'disease_id', 'n_trials', 'status_trials', 'status_drugcentral']].iterrows():
        test_list.append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
        if row['n_trials'] or row['status_trials'] or row['status_drugcentral'] == 1:
            test_label.append(1)
        else:
            test_label.append(0)
    return test_list, test_label
