import click

from ComparisonNRL.create_graph import graph
from ComparisonNRL.nodetovec import nodetovec,HadamardEmbedder,AverageEmbedder,WeightedL1Embedder,WeightedL2Embedder
from ComparisonNRL.pairs import train_pairs,test_pairs
from ComparisonNRL.train import train, validate
from ComparisonNRL.subgraph import subgraph


@click.command(help = 'This cli runs the ComparisonNRL')
@click.argument('nodepath')
@click.argument('edgepath')
@click.argument('featurepath')
@click.argument('graphtype',type=click.Choice(['subgraph','wholegraph','permutation1','permutation2','permutation3','permutation4','permutation5']))
@click.argument('method',type=click.Choice(['node2vec','edge2vec','metapath2vec']))
@click.option('--edgetype',default ='HadamardEmbedder' ,type=click.Choice(['HadamardEmbedder','AverageEmbedder','WeightedL1Embedder','WeightedL2Embedder']))


def main (nodepath,edgepath,featurepath,graphtype,method,edgetype):
    if graphtype == 'subgraph' and method == 'node2vec':
        biggraph = graph(nodepath,edgepath)
        thisgraph,positive_list,positive_label,negative_list,negative_label = subgraph(featurepath,graph,3,10,20)
        model = nodetovec(thisgraph)
        if edgetype == 'HadamardEmbedder':
            positive_vecs = HadamardEmbedder(model,positive_list)
            negative_vecs = HadamardEmbedder(model,negative_list)
        if edgetype ==  'AverageEmbedder':
            positive_vecs = AverageEmbedder(model,positive_list)
            negative_vecs = AverageEmbedder(model,negative_list)
        if edgetype =='WeightedL1Embedder':
            positive_vecs = WeightedL1Embedder(model,positive_list)
            negative_vecs = WeightedL1Embedder(model,negative_list)
        if edgetype == 'WeightedL2Embedder':
            positive_vecs = WeightedL2Embedder(model,positive_list)
            negative_vecs = WeightedL2Embedder(model,negative_list)
        train_vecs = positive_vecs[0:5]+negative_vecs[0:15]
        train_labels = positive_label[0:5] + negative_label [0:15]
        test_vecs = positive_vecs[5:] + negative_vecs[15:]
        test_labels = positive_label[5:] + negative_label [15:]
        lg = train (train_vecs,train_labels)
        roc = validate (test_vecs,test_labels)
        return roc






