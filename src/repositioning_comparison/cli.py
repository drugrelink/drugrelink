# -*- coding: utf-8 -*-

import json
import os
from typing import Type

import click
import joblib
import node2vec.edges

from .create_graph import create_himmelstein_graph
from .download import DIRECTORY, ensure_data
from .nodetovec import embed_average, embed_hadamard, embed_weighted_l1, embed_weighted_l2, fit_node2vec
from .pairs import test_pairs, train_pairs
from .permutation_convert import convert
from .subgraph import generate_subgraph
from .train import train_logistic_regression, validate

DEFAULT_GRAPH_TYPE = 'wholegraph'
GRAPH_TYPES = [
    DEFAULT_GRAPH_TYPE,
    'subgraph',
    'permutation1',
    'permutation2',
    'permutation3',
    'permutation4',
    'permutation5',
]

EMBEDDERS = {
    'hadamard': embed_hadamard,
    'average': embed_average,
    'weighted_l1': embed_weighted_l1,
    'weighted_l2': embed_weighted_l2,
}

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'results'))
RESULTS_DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)


@click.command()
@click.option('-t', '--graph-type', type=click.Choice(GRAPH_TYPES), default=DEFAULT_GRAPH_TYPE)
@click.option('--data-directory', type=click.Path(dir_okay=True, file_okay=False), default=DIRECTORY, show_default=True)
@click.option('-d', '--output-directory', type=click.Path(dir_okay=True, file_okay=False), default=RESULTS_DIRECTORY,
              help='Output directory. Defaults to current working directory.', show_default=True)
@click.option('--method', default='node2vec', type=click.Choice(['node2vec', 'edge2vec', 'metapath2vec']))
@click.option('--embedder', default='hadamard', type=click.Choice(list(EMBEDDERS)))
def main(graph_type: str, data_directory: str, output_directory: str, method: str, embedder: str):
    """This cli runs the ComparisonNRL."""
    nodepath, edgepath, featurepath, validatepath, permutation_paths = ensure_data(directory=data_directory)

    with open(os.path.join(output_directory, 'metadata.txt'), 'w') as file:
        json.dump(
            {
                'graph': graph_type,
                'method': method,
                'embedder': embedder,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    embedder_cls = EMBEDDERS[embedder]

    click.echo(f'Running method={method}, type={graph_type}, embedder={embedder}')
    if graph_type == 'subgraph' and method == 'node2vec':
        subgraph_node2vec_directory = os.path.join(output_directory, 'node2vec_subgraph')
        subgraph_node2vec(nodepath, edgepath, featurepath, embedder_cls, subgraph_node2vec_directory)

    elif graph_type == DEFAULT_GRAPH_TYPE and method == 'node2vec':
        wholegraph_node2vec_directory = os.path.join(output_directory, 'node2vec')
        wholegraph = create_himmelstein_graph(nodepath, edgepath)
        graph_node2vec(wholegraph, wholegraph_node2vec_directory, featurepath, validatepath, embedder_cls)

    elif graph_type == "permutation1" and method == "node2vec":
        graph = convert(permutation_paths[0], 1)


def subgraph_node2vec(
        nodepath,
        edgepath,
        featurepath,
        embedder_cls: Type[node2vec.edges.EdgeEmbedder],
        output_directory,
) -> None:
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    click.echo('creating graph')
    graph = create_himmelstein_graph(nodepath, edgepath)

    click.echo('creating sub-graph')
    (subgraph,
     positive_list,
     positive_labels,
     negative_list,
     negative_labels) = generate_subgraph(featurepath, graph, cutoff=3, pnumber=10, nnumber=20)

    click.echo('fitting node2vec')
    model = fit_node2vec(subgraph)

    click.echo('generating positive and negative vectors')
    negative_vectors = embedder_cls(model, negative_list)
    positive_vectors = embedder_cls(model, positive_list)

    train_vectors = positive_vectors[0:5] + negative_vectors[0:15]
    train_labels = positive_labels[0:5] + negative_labels[0:15]
    train_data = [train_vectors, train_labels]
    with open(os.path.join(output_directory, 'train.json'), 'w') as file:
        json.dump(train_data, file, indent=2, sort_keys=True)

    test_vectors = positive_vectors[5:] + negative_vectors[15:]
    test_labels = positive_labels[5:] + negative_labels[15:]
    test_data = [test_vectors, test_labels]
    with open(os.path.join(output_directory, 'test.json'), 'w') as file:
        json.dump(test_data, file, indent=2, sort_keys=True)

    logistic_regression = train_logistic_regression(train_vectors, train_labels)
    with open(os.path.join(output_directory, 'model.joblib'), 'wb') as file:
        joblib.dump(logistic_regression, file)

    roc = validate(logistic_regression, test_vectors, test_labels)
    with open(os.path.join(output_directory, 'validation.json'), 'w') as file:
        json.dump(
            {
                'ROC': roc,
            },
            file,
            sort_keys=True,
            indent=2,
        )


def graph_node2vec(
        graph,
        output_directory,
        featurepath,
        validatepath,
        embedder_cls: Type[node2vec.edges.EdgeEmbedder],
) -> None:
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    model = fit_node2vec(graph)

    train_list, train_labels = train_pairs(featurepath)
    train_vecs = embedder_cls(model, train_list)
    train_data = [train_vecs, train_labels]
    with open(os.path.join(output_directory, 'train.json'), 'w') as train_file:
        json.dump(train_data, train_file, indent=2, sort_keys=True)

    test_list, test_labels = test_pairs(validatepath)
    test_vecs = embedder_cls(model, test_list)
    test_data = [test_vecs, test_labels]
    with open(os.path.join(output_directory, 'test.json'), 'w') as test_file:
        json.dump(test_data, test_file, indent=2, sort_keys=True)

    lg = train_logistic_regression(train_vecs, train_labels)
    with open(os.path.join(output_directory, 'model.joblib'), 'wb') as model_file:
        joblib.dump(lg, model_file)

    roc = validate(lg, test_vecs, test_labels)
    with open(os.path.join(output_directory, 'validate.txt'), 'w') as validate_file:
        print(f'ROC: {roc}', file=validate_file)
