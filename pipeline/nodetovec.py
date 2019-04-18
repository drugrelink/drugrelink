import node2vec
def nodetovec (graph):
    n_model = node2vec.Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = n_model.fit(window=10, min_count=1, batch_words=4)
    return model
def HadamardEmbedder(model,pair_list):
    edge_vecs = []
    edges_embs = node2vec.edges.HadamardEmbedder(keyed_vectors=model.wv)
    for i in pair_list:
        node1=i[0]
        node2=i[1]
        vec= edges_embs[(node1,node2)].tolist()
        edge_vecs.append(vec)
    return edge_vecs
def AverageEmbedder(model,pair_list):
    edge_vecs = []
    edges_embs = node2vec.edges.AverageEmbedder(keyed_vectors=model.wv)
    for i in pair_list:
        node1=i[0]
        node2=i[1]
        vec= edges_embs[(node1,node2)].tolist()
        edge_vecs.append(vec)
    return edge_vecs 
def WeightedL1Embedder(model,pair_list):
    edge_vecs = []
    edges_embs = node2vec.edges.WeightedL1Embedder(keyed_vectors=model.wv)
    for i in pair_list:
        node1=i[0]
        node2=i[1]
        vec= edges_embs[(node1,node2)].tolist()
        edge_vecs.append(vec)
    return edge_vecs 
def WeightedL2Embedder(model,pair_list):
    edge_vecs = []
    edges_embs = node2vec.edges.WeightedL2Embedder(keyed_vectors=model.wv)
    for i in pair_list:
        node1=i[0]
        node2=i[1]
        vec= edges_embs[(node1,node2)].tolist()
        edge_vecs.append(vec)
    return edge_vecs  