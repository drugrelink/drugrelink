from node2vec import Node2Vec
import node2vec
def node_2vec(graph):
    n_model = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = n_model.fit(window=10, min_count=1, batch_words=4)
    return model

def HadamardEmbedder(model,pair_list):

#This function returns training feature vectors of compound-disease pairs
#para:
#model is the model from embedding method 
#edge_f_type is the way to calculate edge feature          vectors,includingAverageEmbedder,HadamardEmbedder,WeightedL1Embedder,WeightedL2Embedder.
#pair_list is compound-disease list with labels known

     
    edge_vecs = []
    edges_embs = node2vec.edges.HadamardEmbedder(keyed_vectors=model.wv)
    for i in pair_list:
        node1=i[0]
        node2=i[1]
        vec= edges_embs[(node1,node2)].tolist()
        edge_vecs.append(vec)
    return edge_vecs

def test_data(model,edge_f_type,pair_list):

#This function returns testing feature vectors of compound-disease pairs
#para:
#model is the model from embedding method 
#edge_f_type is the way to calculate edge feature          vectors,includingAverageEmbedder,HadamardEmbedder,WeightedL1Embedder,WeightedL2Embedder.
#pair_list is compound-disease list with labels known'''
    test_x = []
    edges_embs = node2vec.edges.edge_f_type(keyed_vectors=model.wv)
    for i in pair_list:
        node1=i[0]
        node2=i[1]
        edge_vec= edges_embs[(node1,node2)].tolist()
        test_x.append(edge_vec)
    return test_x    