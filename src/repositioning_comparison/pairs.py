# -*- coding: utf-8 -*-

import bz2

import pandas as pd


def train_pairs(path):
    f = bz2.BZ2File(path, 'r')
    df_features = pd.read_csv(f, sep='\t', low_memory=False)

    train_list = []
    train_label = []
    for _, row in df_features[['compound_id', 'disease_id', 'status']].iterrows():
        train_list.append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
        train_label.append(row['status'])
    return train_list, train_label


def test_pairs(validationpath, trainpath):
    df_validate = pd.read_csv(validationpath, sep='\t')
    diseaseModifying_pairs, diseaseModifying_labels = train_pairs(trainpath)
    diseaseModifying =[diseaseModifying_pairs,diseaseModifying_labels ]
    clinicalTrials = [[],[]]
    drugCentral = [[],[]]
    for _, row in df_validate[
        ['compound_id', 'disease_id', 'n_trials', 'status_trials', 'status_drugcentral']].iterrows():
        if row['status_trials'] ==1:
            clinicalTrials[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            clinicalTrials[1].append(1)
        if row['status_drugcentral'] == 1:
            drugCentral[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            drugCentral[1].append(1)
    return diseaseModifying, clinicalTrials, drugCentral













    return test_list, test_label

