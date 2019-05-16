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


def test_pairs(
        *,
        validation_path,
        train_path,
        symptomatic_path,
):
    """"""
    df_validate = pd.read_csv(validation_path, sep='\t')
    df_symptomatic = pd.read_csv(symptomatic_path, sep='\t')
    disease_modifying_pairs, disease_modifying_labels = train_pairs(train_path)
    disease_modifying = [disease_modifying_pairs, disease_modifying_labels]
    clinical_trials = [[], []]
    drug_central = [[], []]
    symptomatic = [[], []]

    for _, row in df_validate[['compound_id', 'disease_id', 'status_trials', 'status_drugcentral']].iterrows():
        clinical_trials[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
        if row['status_trials'] == 1:
            clinical_trials[1].append(1)
        elif row['status_trials'] == 0:
            clinical_trials[1].append(0)

        drug_central[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
        if row['status_drugcentral'] == 1:
            drug_central[1].append(1)
        elif row['status_drugcentral'] == 0:
            drug_central[1].append(0)

    for _, row in df_symptomatic[['compound_id', 'disease_id', 'category']].iterrows():

        if row['category'] == 'SYM':
            symptomatic[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            symptomatic[1].append(1)
        elif row['category'] != 'DM':
            symptomatic[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            symptomatic[1].append(0)

    return disease_modifying, clinical_trials, drug_central, symptomatic
