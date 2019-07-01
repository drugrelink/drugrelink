"""Prepare the edgelist for edge2vec. 4 columns, source, target, edge type, edge id."""
import numpy as np
import pandas as pd
import gzip
import logging
logger = logging.getLogger(__name__)


def prepare_edge2vec(edge_path, save_path) -> None:
    """
    :param edge_path: edge_path of the file of edges downloaded from Himmelstein's integrate repository:
    :param save_path: A path to save the new edge file which is suitable for edge2vec.
    :return: None
    """
    with gzip.open(edge_path,'rb') as file:
        edge_df = pd.read_csv(file, sep='\t')
    ser = edge_df['metaedge'].value_counts()
    name_list = ser.keys()
    metaedge_map={}
    for i in range(len(name_list)):
        metaedge_map[name_list[i]]=i+1
    for _,row in edge_df.iterrows():
        row['metaedge'] = metaedge_map[row['metaedge']]
        if 'Biological Process' in row['source']:
            row['source'] = row['source'].replace('Biological Process','BiologicalProcess')
        if 'Molecular Function' in row['source']:
            row['source'] = row['source'].replace('Molecular Function','MolecularFunction')
        if 'Cellular Component' in row['source']:
            row['source'] = row['source'].replace('Cellular Component','CellularComponent')
        if 'Pharmacologic Class' in row['source']:
            row['source'] = row['source'].replace('Pharmacologic Class','PharmacologicClass')
        if 'Side Effect' in row['source']:
            row['source'] = row['source'].replace('Side Effect','SideEffect')
        if 'Biological Process' in row['target']:
            row['target'] = row['target'].replace('Biological Process','BiologicalProcess')
        if 'Molecular Function' in row['target']:
            row['target'] = row['target'].replace('Molecular Function','MolecularFunction')
        if 'Cellular Component' in row['target']:
            row['target'] = row['target'].replace('Cellular Component','CellularComponent')
        if 'Pharmacologic Class' in row['target']:
            row['target'] = row['target'].replace('Pharmacologic Class','PharmacologicClass')
        if 'Side Effect' in row['target']:
            row['target'] = row['target'].replace('Side Effect','SideEffect')
    a = len(edge_df)
    id_list=list(range(1,a+1))
    edge_df.insert(loc=3, column='id', value=id_list)
    column = ['source','target','metaedge','id']
    edge_df=edge_df.reindex(columns=column)
    edge_df.to_csv(save_path, header=None, index=None, sep=' ')


