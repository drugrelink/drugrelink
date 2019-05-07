# -*- coding: utf-8 -*-

import json
import logging
import os
import pickle
import sys

import click
import joblib

from .create_graph import create_himmelstein_graph
from .download import DIRECTORY, ensure_data
from .embedders import EMBEDDERS
from .node2vec_utils import fit_node2vec
from .pairs import test_pairs, train_pairs
from .permutation_convert import convert
from .subgraph import generate_subgraph
from .train import train_logistic_regression, validate

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_TYPE = 'subgraph'
GRAPH_TYPES = [
    'graph',
    'subgraph',
    'permutation1',
    'permutation2',
    'permutation3',
    'permutation4',
    'permutation5',
]

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'results'))
RESULTS_DIRECTORY = os.environ.get('REPOSITIONING_COMPARISON_DIRECTORY', DEFAULT_DIRECTORY)


@click.command()
@click.option('-t', '--graph-type', type=click.Choice(GRAPH_TYPES), default=DEFAULT_GRAPH_TYPE, show_default=True)
@click.option('--data-directory', type=click.Path(dir_okay=True, file_okay=False), default=DIRECTORY, show_default=True)
@click.option('-d', '--output-directory', type=click.Path(dir_okay=True, file_okay=False), default=RESULTS_DIRECTORY,
              help='Output directory. Defaults to current working directory.', show_default=True)
@click.option('--method', default='node2vec', type=click.Choice(['node2vec', 'edge2vec', 'metapath2vec']),
              show_default=True)
@click.option('--embedder', default='hadamard', type=click.Choice(list(EMBEDDERS)), show_default=True)
def main(graph_type: str, data_directory: str, output_directory: str, method: str, embedder: str):
    """This cli runs the ComparisonNRL."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.setLevel(logging.INFO)

    node_path, edge_path, feature_path, validate_path, permutation_paths = ensure_data(directory=data_directory)

    # TODO add random seed as argument

    subdirectory = os.path.join(output_directory, f'{method}_{graph_type}_{embedder}')
    os.makedirs(subdirectory, exist_ok=True)
    click.echo(f'Running method={method}, type={graph_type}, embedder={embedder}')
    if method == 'node2vec':
        if graph_type == 'subgraph':
            run_node2vec_subgraph(
                node_path=node_path,
                edge_path=edge_path,
                feature_path=feature_path,
                embedder=embedder,
                output_directory=subdirectory,
            )

        elif graph_type == 'graph':
            run_node2vec(
                node_path=node_path,
                edge_path=edge_path,
                feature_path=feature_path,
                validate_path=validate_path,
                embedder=embedder,
                output_directory=subdirectory,
            )

        elif graph_type == "permutation1":
            graph = convert(permutation_paths[0], 1)

        else:
            click.secho(f'Graph type not implemented yet: {graph_type}')
            sys.exit(1)

    else:
        click.secho(f'Method not implemented yet: {method}')
        sys.exit(1)


def run_node2vec_subgraph(
        *,
        node_path,
        edge_path,
        feature_path,
        embedder: str,
        output_directory,
        n_train_positive: int = 5,
        n_train_negative: int = 15,
) -> None:
    # Define some output paths
    metadata_json_path = os.path.join(output_directory, 'metadata.json')
    transition_probabilities_path = os.path.join(output_directory, 'transition_probabilities.json')

    with open(metadata_json_path, 'w') as file:
        json.dump(
            {
                'graph': 'subgraph',
                'method': 'node2vec',
                'embedder': embedder,
                'n_train_positive': n_train_positive,
                'n_train_negative': n_train_negative,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    subgraph_path = os.path.join(output_directory, 'subgraph.pickle')
    positive_list_path = os.path.join(output_directory, 'positive_list.pickle')
    positive_labels_path = os.path.join(output_directory, 'positive_labels.pickle')
    negative_list_path = os.path.join(output_directory, 'negative_list.pickle')
    negative_labels_path = os.path.join(output_directory, 'negative_labels.pickle')

    if os.path.exists(subgraph_path):
        logger.info('loading pickled subgraph info')
        with open(subgraph_path, 'rb') as file:
            subgraph = pickle.load(file)
        with open(positive_list_path, 'rb') as file:
            positive_list = pickle.load(file)
        with open(positive_labels_path, 'rb') as file:
            positive_labels = pickle.load(file)
        with open(negative_list_path, 'rb') as file:
            negative_list = pickle.load(file)
        with open(negative_labels_path, 'rb') as file:
            negative_labels = pickle.load(file)
        logger.info('loaded pickled subgraph info')

    else:
        click.echo('creating graph')
        graph = create_himmelstein_graph(node_path, edge_path)

        click.echo('creating sub-graph')
        (subgraph,
         positive_list,
         positive_labels,
         negative_list,
         negative_labels) = generate_subgraph(
            feature_path,
            graph,
            max_simple_path_length=3,
            n_positive=10,  # TODO calculate positive and negative number based on n_train_positive
            n_negative=20,
        )

        logger.info('dumping pickled subgraph info')
        with open(subgraph_path, 'wb') as file:
            pickle.dump(subgraph, file, protocol=-1)
        with open(positive_list_path, 'wb') as file:
            pickle.dump(positive_list, file, protocol=-1)
        with open(positive_labels_path, 'wb') as file:
            pickle.dump(positive_labels, file, protocol=-1)
        with open(negative_list_path, 'wb') as file:
            pickle.dump(negative_list, file, protocol=-1)
        with open(negative_labels_path, 'wb') as file:
            pickle.dump(negative_labels, file, protocol=-1)

    click.echo('fitting node2vec/word2vec')
    model = fit_node2vec(subgraph, transition_probabilities_path=transition_probabilities_path)

    click.echo('saving word2vec')
    model.save(os.path.join(output_directory, 'word2vec_model.pickle'))

    click.echo('generating vectors')
    embedder_function = EMBEDDERS[embedder]
    positive_vectors = embedder_function(model, positive_list)
    negative_vectors = embedder_function(model, negative_list)

    train_vectors = positive_vectors[:n_train_positive] + negative_vectors[:n_train_negative]
    train_labels = positive_labels[:n_train_positive] + negative_labels[:n_train_negative]
    test_vectors = positive_vectors[n_train_positive:] + negative_vectors[n_train_negative:]
    test_labels = positive_labels[n_train_positive:] + negative_labels[n_train_negative:]

    _train_evaluate_generate_artifacts(output_directory, train_vectors, train_labels, test_vectors, test_labels)


def run_node2vec(
        *,
        node_path,
        edge_path,
        feature_path,
        validate_path,
        output_directory,
        embedder: str,
) -> None:
    transition_probability_path = os.path.join(output_directory, 'transition_probabilities.json')

    graph = create_himmelstein_graph(node_path, edge_path)
    model = fit_node2vec(graph, transition_probabilities_path=transition_probability_path)

    embedder_function = EMBEDDERS[embedder]
    train_list, train_labels = train_pairs(feature_path)
    #  TODO why build multiple embedders separately and not single one then split vectors after the fact?
    train_vectors = embedder_function(model, train_list)
    test_list, test_labels = test_pairs(validate_path)
    test_vectors = embedder_function(model, test_list)

    _train_evaluate_generate_artifacts(output_directory, train_vectors, train_labels, test_vectors, test_labels)


def _train_evaluate_generate_artifacts(
        output_directory,
        train_vectors,
        train_labels,
        test_vectors,
        test_labels,
) -> None:
    with open(os.path.join(output_directory, 'train.json'), 'w') as file:
        json.dump(
            [
                dict(vector=train_vector, label=train_label)
                for train_vector, train_label in zip(train_vectors, train_labels)
            ],
            file,
            indent=2,
            sort_keys=True,
        )

    with open(os.path.join(output_directory, 'test.json'), 'w') as file:
        json.dump(
            [
                dict(vector=train_vector, label=train_label)
                for train_vector, train_label in zip(test_vectors, test_labels)
            ],
            file,
            indent=2,
            sort_keys=True,
        )

    click.echo('training logistic regression classifier')
    logistic_regression = train_logistic_regression(train_vectors, train_labels)
    with open(os.path.join(output_directory, 'logistic_regression_clf.joblib'), 'wb') as file:
        joblib.dump(logistic_regression, file)

    click.echo('validating logistic regression classifier')
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
