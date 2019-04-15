
def train_pairs():
    path_pairs = str(Path().absolute())
    df_features = pd.read_csv(path_node+'/original_graph_data/transformed-features.tsv',sep='\t')
    train_list=[]
    train_label=[]
    for _,row in df_features[['compound_id','disease_id','status']].iterrows():
        train_list.append(['Compound::'+row['compound_id'],'Disease::'+row['disease_id']])
        train_label.append(row['status'])
    return train_list,train_label
def test_pairs():
    path_pairs = str(Path().absolute())
    df_features = pd.read_csv(path_node+'/original_graph_data/transformed-features.tsv',sep='\t')
    test_list=[]
    test_label=[]
    for _,row in df_features[['compound_id','disease_id','status']].iterrows():
        test_list.append(['Compound::'+row['compound_id'],'Disease::'+row['disease_id']])
        test_label.append(row['status'])
    return test_list,test_label    