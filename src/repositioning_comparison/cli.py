import os

import click
import simplejson
from joblib import dump, load

from .create_graph import create_himmelstein_graph
from .download import ensure_data, DIRECTORY
from .nodetovec import nodetovec, HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
from .pairs import train_pairs, test_pairs
from .subgraph import subgraph
from .train import train, validate

GRAPH_TYPES = ['subgraph', 'wholegraph', 'permutation1', 'permutation2', 'permutation3', 'permutation4', 'permutation5']
EMBEDDER_TYPES = ['HadamardEmbedder', 'AverageEmbedder', 'WeightedL1Embedder', 'WeightedL2Embedder']
HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'results'))
RESULTS_DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)

@click.command()
@click.argument('graphtype', type=click.Choice(GRAPH_TYPES))
@click.option('--data-directory', type=click.Path(dir_okay=True, file_okay=False), default=DIRECTORY, show_default=True)
@click.option('-d', '--output-directory', type=click.Path(dir_okay=True, file_okay=False), default=RESULTS_DIRECTORY,
              help='Output directory. Defaults to current working directory.')
@click.option('--method', default='node2vec', type=click.Choice(['node2vec', 'edge2vec', 'metapath2vec']))
@click.option('--embedder', default='HadamardEmbedder', type=click.Choice(EMBEDDER_TYPES))
def main(graphtype, data_directory, output_directory, method, embedder):
    """This cli runs the ComparisonNRL."""
    nodepath, edgepath, featurepath, validatepath, permutation_paths = ensure_data(directory=data_directory)

    if graphtype == 'subgraph' and method == 'node2vec':
        result = subgraph_node2vec(nodepath, edgepath, featurepath, embedder,output_directory)
        click.echo(f'Result: {result}')

    elif graphtype == 'wholegraph' and method == 'node2vec':
        wholegraph = create_himmelstein_graph(nodepath,edgepath)
        result = graph_node2vec(wholegraph,nodepath, edgepath, featurepath,validatepath, embedder)
        click.echo(f'Result: {result}')


def subgraph_node2vec(nodepath, edgepath, featurepath, embedder,output_directory):
    biggraph = create_himmelstein_graph(nodepath, edgepath)
    thisgraph, positive_list, positive_label, negative_list, negative_label = subgraph(featurepath, biggraph, 3, 10,20)
    dir_path =  os.path.join (output_directory,'subgraph_node2vec')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    metadata_file = os.path.join(dir_path,'metadata')
    if not os.path.exists(metadata_file):
        metadata_file = open(metadata_file,'w')
    metadata_file = open(metadata_file,'w')
    metadata_file.write('embedder: '+embedder+ '\n')
    metadata_file.close()

    train_file = os.path.join(dir_path,'train')
    if not os.path.exists(train_file):
        train_file = open(train_file,'w')
    train_file = open(train_file,'w')

    test_file = os.path.join(dir_path,'test')
    if not os.path.exists(test_file):
        test_file = open(test_file,'w')
    test_file = open(test_file,'w')

    model_file = os.path.join(dir_path,'model')
    if not os.path.exists(model_file):
        model_file = open(model_file,'wb')
    model_file = open(model_file,'w')

    validate_file = os.path.join(dir_path,'validate')
    if not os.path.exists(model_file):
        validate_file = open(validate_file,'w')
    validate_file = open(validate_file,'w')

    model = nodetovec(thisgraph)
    if embedder == 'HadamardEmbedder':
        positive_vecs = HadamardEmbedder(model, positive_list)
        negative_vecs = HadamardEmbedder(model, negative_list)
    elif embedder == 'AverageEmbedder':
        positive_vecs = AverageEmbedder(model, positive_list)
        negative_vecs = AverageEmbedder(model, negative_list)
    elif embedder == 'WeightedL1Embedder':
        positive_vecs = WeightedL1Embedder(model, positive_list)
        negative_vecs = WeightedL1Embedder(model, negative_list)
    elif embedder == 'WeightedL2Embedder':
        positive_vecs = WeightedL2Embedder(model, positive_list)
        negative_vecs = WeightedL2Embedder(model, negative_list)
    else:
        raise NotImplementedError

    train_vecs = positive_vecs[0:5] + negative_vecs[0:15]
    train_labels = positive_label[0:5] + negative_label[0:15]
    train_data = [train_vecs,train_labels]
    simplejson.dump(train_data, train_file)
    train_file.close()

    test_vecs = positive_vecs[5:] + negative_vecs[15:]
    test_labels = positive_label[5:] + negative_label[15:]
    test_data = [test_vecs , test_labels]
    simplejson.dump(test_data,test_file)
    test_file.close()

    lg = train(train_vecs, train_labels)
    dump(lg, model_file)
    roc = validate(lg, test_vecs, test_labels)
    validate_file.write(roc)




def graph_node2vec(graph,nodepath, edgepath, featurepath,validatepath, embedder):
    model = nodetovec(graph)
    train_list,train_labels = train_pairs(featurepath)
    test_list,test_labels = test_pairs(validatepath)
    if embedder == 'HadamardEmbedder':
        train_vecs = HadamardEmbedder(model,train_list)
        test_vecs = HadamardEmbedder(model,test_list)
    elif embedder == 'AverageEmbedder':
        train_vecs = AverageEmbedder(model,train_list)
        test_vecs = AverageEmbedder(model,test_list)
    elif embedder == 'WeightedL1Embedder':
        train_vecs = WeightedL1Embedder(model,train_list)
        test_vecs = WeightedL1Embedder(model,test_list)
    elif embedder == 'WeightedL2Embedder':
        train_vecs = WeightedL2Embedder(model,train_list)
        test_vecs = WeightedL2Embedder(model,test_list)
    else:
        raise NotImplementedError
    lg = train(train_vecs, train_labels)
    roc = validate(lg, test_vecs, test_labels)
    return roc
