import numpy as np
import pandas as pd
import gzip
import logging
logger = logging.getLogger(__name__)


def prepare_edge2vec(edge_path, save_path):
    with gzip.open(edge_path,'rb') as file:
        edge_df = pd.read_csv(file, sep='\t')
    ser = edge_df['metaedge'].value_counts()
    name_list = ser.keys()
    metaedge_map={}
    for i in range(len(name_list)):
        metaedge_map[name_list[i]]=i+1
    for _,row in edge_df.iterrows():
        row['metaedge'] = metaedge_map[row['metaedge']]
    a = len(edge_df)
    id_list=list(range(1,a+1))
    edge_df.insert(loc=3, column='id', value=id_list)
    column = ['source','target','metaedge','id']
    edge_df=edge_df.reindex(columns=column)
    edge_df.to_csv(save_path, header=None, index=None, sep=' ')


