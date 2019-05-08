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


def test_pairs(validation_path, train_path, symptomatic_path):
    df_validate = pd.read_csv(validation_path, sep='\t')
    df_symptomatic = pd.read_csv(symptomatic_path, sep='\t')
    diseaseModifying_pairs, diseaseModifying_labels = train_pairs(train_path)
    diseaseModifying =[diseaseModifying_pairs,diseaseModifying_labels ]
    clinicalTrials = [[],[]]
    drugCentral = [[],[]]
    symptomatic = [[],[]]
    for _, row in df_validate[
        ['compound_id', 'disease_id', 'status_trials', 'status_drugcentral']].iterrows():
        if row['status_trials'] ==1:
            clinicalTrials[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            clinicalTrials[1].append(1)
        elif row['status_trials'] == 0:
            clinicalTrials[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            clinicalTrials[1].append(0)
        if row['status_drugcentral'] == 1:
            drugCentral[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            drugCentral[1].append(1)
        elif row['status_drugcentral'] ==0:
            drugCentral[0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            drugCentral[1].append(0)
    for _, row in df_symptomatic[['compound_id', 'disease_id','category']].iterrows():
        if row['category'] == 'SYM':
            symptomatic [0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            symptomatic[1].append(1)
        else:
            symptomatic [0].append(['Compound::' + row['compound_id'], 'Disease::' + row['disease_id']])
            symptomatic[1].append(0)



    return diseaseModifying,clinicalTrials,drugCentral,symptomatic













    return test_list, test_label

