# -*- coding: utf-8 -*-

import bz2
import os

import pandas as pd
from .download import DATA_DIRECTORY
from .embedders import get_embedder
from node2vec.edges import HadamardEmbedder
import numpy as np

def train_test_pairs(
        *,
        validation_path,
        train_path,
        symptomatic_path,
) -> [tuple]:
    """
    This function is to generate training and testing drug-disease pairs and respective labels.
    :param validation_path: path for validation-statuses.tsv
    :param train_path: path for transformed-features.tsv.bz2
    :param symptomatic_path: path for probabilities.tsv
    :return: lists of traning and testing pairs with labels
            disease_modifying_training: training data for Rephetio
            disease_modifying_testing: disease modify test data
            clinical_trials: clinical_trials test data
            drug_central: drug_central test data
            symptomatic: symptomatic test data
    """
    df_validate = pd.read_csv(validation_path, sep='\t')
    df_symptomatic = pd.read_csv(symptomatic_path, sep='\t')
    f_feature = bz2.BZ2File(train_path, 'r')
    df_features = pd.read_csv(f_feature, sep='\t', low_memory=False)

    disease_modifying_training = []
    clinical_trials = []
    drug_central = []
    symptomatic_raw = []
    disease_modifying_raw = []
    non_status_raw = []
    symptomatic_for_DM = []
    disease_modifying_testing =[]
    symptomatic = []

    for _, row in df_features[['compound_id', 'disease_id', 'status']].iterrows():
        compound = 'Compound::' + row['compound_id']
        disease = 'Disease::' + row['disease_id']
        status = row['status']
        disease_modifying_training.append(tuple((
            compound,
            disease,
            status
        )))

    for _, row in df_validate[['compound_id', 'disease_id', 'status_trials', 'status_drugcentral']].iterrows():
        compound = 'Compound::' + row['compound_id']
        disease = 'Disease::' + row['disease_id']
        status = row['status_trials']
        clinical_trials.append(tuple((
            compound,
            disease,
            status
        )))
        status = row['status_drugcentral']

        drug_central.append(tuple((
            compound,
            disease,
            status
        )))

    for _, row in df_symptomatic[['compound_id', 'disease_id', 'category']].iterrows():
        compound = 'Compound::' + row['compound_id']
        disease = 'Disease::' + row['disease_id']
        if row['category'] == 'SYM':
            symptomatic_raw.append(tuple((
                compound,
                disease,
                1
            )))
            symptomatic_for_DM.append(tuple((
                compound,
                disease,
                0
            )))
        elif row['category'] == 'DM':
            disease_modifying_raw.append(tuple((
                compound,
                disease,
                1
            )))
        else:
            non_status_raw.append(tuple((
                compound,
                disease,
                0
            ))

            )
            symptomatic = symptomatic_raw + non_status_raw
            disease_modifying_testing = disease_modifying_raw + symptomatic_for_DM + non_status_raw
    return np.array(disease_modifying_training), np.array(disease_modifying_testing), np.array(clinical_trials), np.array(drug_central), np.array(symptomatic)

def data_non_overlap(
        *,
        validation_path,
        train_path,
        symptomatic_path,
        output_directory: str = None
):
    """
    Generate a df of data with non overlap.
    :param validation_path:
    :param train_path:
    :param symptomatic_path:
    :param output_directory:
    :return: a dataframe of all drug-disease pairs with non-overlap.
    """
    if not output_directory:
        output_directory = os.path.join(DATA_DIRECTORY, 'data_non_overlap')
    else:
        output_directory = output_directory

    if output_directory and os.path.exists(output_directory):
        data_df = pd.read_csv(output_directory, sep=',')
    else:

        _, disease_modifying, clinical_trials, drug_central, symptomatic = train_test_pairs(validation_path=validation_path,
                                                                                   train_path=train_path,
                                                                                   symptomatic_path=symptomatic_path)
        rows = disease_modifying + clinical_trials + drug_central + symptomatic
        data_df = pd.DataFrame(rows, columns=['compound', 'disease', 'label'])
        data_df.drop_duplicates(keep='first', inplace=True)
        data_df.sort_values(by='label')
        data_df.drop_duplicates(subset=['compound', 'disease'], keep='last', inplace=True)
        pd.DataFrame.to_csv(data_df, output_directory)

    return data_df


def pairs_vectors(df,word2vec):
    """
    Generate vectors for drug-disease pairs
    :param df: a dataframe of drug-disease pairs
    :param word2vec: trained word2vec model
    :return: list of vectors
    """
    vectors=[]
    for _,row in df[['compound','disease']].iterrows():
        c=row['compound']
        d=row['disease']
        edges_embs = HadamardEmbedder(keyed_vectors=word2vec.wv)
        vector = edges_embs[(c,d)]
        vectors.append(vector)
    return vectors
