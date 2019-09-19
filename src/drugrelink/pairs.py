# -*- coding: utf-8 -*-

import bz2
import os

import pandas as pd
from .download import DATA_DIRECTORY
from .embedders import get_embedder
from node2vec.edges import HadamardEmbedder

def test_pairs(
        *,
        validation_path,
        train_path,
        symptomatic_path,
):
    """"""
    df_validate = pd.read_csv(validation_path, sep='\t')
    df_symptomatic = pd.read_csv(symptomatic_path, sep='\t')
    f = bz2.BZ2File(train_path, 'r')
    df_features = pd.read_csv(f, sep='\t', low_memory=False)
    disease_modifying = []
    clinical_trials = []
    drug_central = []
    symptomatic = []
    for _, row in df_features[['compound_id', 'disease_id', 'status']].iterrows():
        compound = 'Compound::' + row['compound_id']
        disease = 'Disease::' + row['disease_id']
        status = row['status']
        disease_modifying.append(tuple((
            compound,
            disease,
            status
        )))

    for _, row in df_validate[['compound_id', 'disease_id', 'status_trials', 'status_drugcentral']].iterrows():
        compound = 'Compound::' + row['compound_id']
        disease = 'Disease::' + row['disease_id']
        status = row['status_trials']
        t = tuple()
        clinical_trials.append(tuple((
            compound,
            disease,
            status
        )))
        status = row['status_drugcentral']

        clinical_trials.append(tuple((
            compound,
            disease,
            status
        )))

    for _, row in df_symptomatic[['compound_id', 'disease_id', 'category']].iterrows():
        compound = 'Compound::' + row['compound_id']
        disease = 'Disease::' + row['disease_id']
        if row['category'] == 'SYM':
            symptomatic.append(tuple((
                compound,
                disease,
                1
            )))
        elif row['category'] != 'DM':
            symptomatic.append(tuple((
                compound,
                disease,
                0
            )))
    return disease_modifying, clinical_trials, drug_central, symptomatic


def data_non_overlap(
        *,
        validation_path,
        train_path,
        symptomatic_path,
        output_directory: str = None
):
    """
    Return a df of data with non overlap.
    :param validation_path:
    :param train_path:
    :param symptomatic_path:
    :param output_directory:
    :return:
    """
    if not output_directory:
        output_directory = os.path.join(DATA_DIRECTORY, 'data_non_overlap')
    else:
        output_directory = output_directory

    if output_directory and os.path.exists(output_directory):
        data_df = pd.read_csv(output_directory, sep=',')
    else:

        disease_modifying, clinical_trials, drug_central, symptomatic = test_pairs(validation_path=validation_path,
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
    vectors=[]
    for _,row in df[['compound','disease']].iterrows():
        c=row['compound']
        d=row['disease']
        edges_embs = HadamardEmbedder(keyed_vectors=word2vec.wv)
        vector = edges_embs[(c,d)]
        vectors.append(vector)
    return vectors
