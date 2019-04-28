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
from .permutation_convert import convert

GRAPH_TYPES = ['subgraph', 'wholegraph', 'permutation1', 'permutation2', 'permutation3', 'permutation4', 'permutation5']
EMBEDDERS = {
    'HadamardEmbedder': HadamardEmbedder,
    'AverageEmbedder':AverageEmbedder,
    'WeightedL1Embedder':  WeightedL1Embedder,
    'WeightedL2Embedder': WeightedL2Embedder,
}


HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'results'))
RESULTS_DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)


@click.command()
@click.argument('graphtype', type=click.Choice(GRAPH_TYPES))
@click.option('--data-directory', type=click.Path(dir_okay=True, file_okay=False), default=DIRECTORY, show_default=True)
@click.option('-d', '--output-directory', type=click.Path(dir_okay=True, file_okay=False), default=RESULTS_DIRECTORY,
              help='Output directory. Defaults to current working directory.', show_default=True)
@click.option('--method', default='node2vec', type=click.Choice(['node2vec', 'edge2vec', 'metapath2vec']))
@click.option('--embedder', default='HadamardEmbedder', type=click.Choice(list(EMBEDDERS)))
def main(graphtype, data_directory, output_directory, method, embedder):
    """This cli runs the ComparisonNRL."""
    nodepath, edgepath, featurepath, validatepath, permutation_paths = ensure_data(directory=data_directory)

    if graphtype == 'subgraph' and method == 'node2vec':
        data = metadata(graphtype, method, embedder)
        subgraph_node2vec_directory = os.path.join(output_directory, 'subgraph_node2vec')
        result = subgraph_node2vec(nodepath, edgepath, featurepath, embedder, subgraph_node2vec_directory, data)
        click.echo(f'Result: {result}')

    elif graphtype == 'wholegraph' and method == 'node2vec':
        wholegraph_node2vec_directory = os.path.join(output_directory, 'wholegraph_node2vec')
        wholegraph = create_himmelstein_graph(nodepath, edgepath)
        data = metadata(graphtype, method, embedder)
        result = graph_node2vec(wholegraph, wholegraph_node2vec_directory,featurepath, validatepath, embedder, data)
        click.echo(f'Result: {result}')
    elif graphtype == "permutation1" and method == "node2vec":
        graph = convert(permutation_paths[0],1)
        return 0



def subgraph_node2vec(nodepath, edgepath, featurepath, embedder, output_directory, data):
    biggraph = create_himmelstein_graph(nodepath, edgepath)
    thisgraph, positive_list, positive_label, negative_list, negative_label = subgraph(featurepath, biggraph, 3, 10, 20)


    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    with open(os.path.join(output_directory, 'metadata.txt'), 'w') as metadata_file:
        metadata_file.write(data)
    metadata_file.close()

    model = nodetovec(thisgraph)

    embedder_cls = EMBEDDERS[embedder]
    positive_vecs = embedder_cls(model, positive_list)
    negative_vecs = embedder_cls(model, negative_list)

    train_vecs = positive_vecs[0:5] + negative_vecs[0:15]
    train_labels = positive_label[0:5] + negative_label[0:15]
    train_data = [train_vecs, train_labels]
    with open(os.path.join(output_directory, 'train.json'), 'w') as train_file:
        simplejson.dump(train_data, train_file)
    train_file.close()

    test_vecs = positive_vecs[5:] + negative_vecs[15:]
    test_labels = positive_label[5:] + negative_label[15:]
    test_data = [test_vecs, test_labels]
    with open(os.path.join(output_directory, 'test.json'), 'w') as test_file:
        simplejson.dump(test_data, test_file)
    test_file.close()

    lg = train(train_vecs, train_labels)
    with open(os.path.join(output_directory, 'model.joblib'), 'wb') as model_file:
        dump(lg, model_file)
    model_file.close()

    roc = validate(lg, test_vecs, test_labels)
    with open(os.path.join(output_directory, 'validate.txt'), 'w') as validate_file:
        print(f'ROC: {roc}', file=validate_file)
    validate_file.close()


def graph_node2vec(graph, output_directory,featurepath, validatepath, embedder, data):
    model = nodetovec(graph)
    train_list, train_labels = train_pairs(featurepath)
    test_list, test_labels = test_pairs(validatepath)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    with open(os.path.join(output_directory, 'metadata.txt'), 'w') as metadata_file:
        metadata_file.write(data)
    metadata_file.close()

    embedder_cls = EMBEDDERS[embedder]
    train_vecs = embedder_cls(model, train_list)
    test_vecs = embedder_cls(model, test_list)

    train_data = [train_vecs, train_labels]
    with open(os.path.join(output_directory, 'train.json'), 'w') as train_file:
        simplejson.dump(train_data, train_file)
    train_file.close()

    test_data = [test_vecs, test_labels]
    with open(os.path.join(output_directory, 'test.json'), 'w') as test_file:
        simplejson.dump(test_data, test_file)
    test_file.close()


    lg = train(train_vecs, train_labels)
    with open(os.path.join(output_directory, 'model.joblib'), 'wb') as model_file:
        dump(lg, model_file)
    model_file.close()

    roc = validate(lg, test_vecs, test_labels)
    with open(os.path.join(output_directory, 'validate.txt'), 'w') as validate_file:
        print(f'ROC: {roc}', file=validate_file)
    validate_file.close()




def metadata(graphtype, method, embedder):
    return 'graph: ' + graphtype + '\n'+ 'method: ' + method + '\n'+ 'embdder: ' + embedder + '\n'
