# -*- coding: utf-8 -*-

import datetime
import json
import logging
import os
import pickle
import sys
from typing import Optional

import click
import joblib

from .create_graph import create_himmelstein_graph
from .download import ensure_data
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
@click.option('--data-directory', type=click.Path(dir_okay=True, file_okay=False))
@click.option('-d', '--output-directory', type=click.Path(dir_okay=True, file_okay=False), default=RESULTS_DIRECTORY,
              help='Output directory. Defaults to current working directory.', show_default=True)
@click.option('--config', type=click.File())
@click.option('--method', default='node2vec', type=click.Choice(['node2vec', 'edge2vec', 'metapath2vec']),
              show_default=True)
@click.option('--embedder', default='hadamard', type=click.Choice(list(EMBEDDERS)), show_default=True)
def main(graph_type: str, data_directory: str, output_directory: str, config: str, method: str, embedder: str):
    """This cli runs the ComparisonNRL."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.setLevel(logging.INFO)

    if config is not None:
        config = json.load(config)
        return run_graph_node2vec_from_config(**config)

    (node_path, edge_path, feature_path, validate_path,
     symptomatic_path, permutation_paths) = ensure_data(directory=data_directory)

    # TODO add random seed as argument
    #currentDT = datetime.datetime.now()
    #ct = str(currentDT)[0:16]
    #subdirectory = os.path.join(output_directory, f'{method}_{graph_type}_{embedder}_{ct}')
    #os.makedirs(subdirectory, exist_ok=True)
    click.echo(f'Running method={method}, type={graph_type}, embedder={embedder}')
    if method == 'node2vec':
        if graph_type == 'subgraph':
            run_node2vec_subgraph(
                node_path=node_path,
                edge_path=edge_path,
                feature_path=feature_path,
                embedder=embedder,
                output_directory=None,
            )

        elif graph_type == 'graph':
            raise RuntimeError('Use config dict with --config flag instead')

        elif graph_type == "permutation1":
            graph = convert(permutation_paths[0], 1)

        else:
            click.secho(f'Graph type not implemented yet: {graph_type}')
            sys.exit(1)

    else:
        click.secho(f'Method not implemented yet: {method}')
        sys.exit(1)


def run_graph_node2vec_from_config(
        embedder: str,
        dimensions: int,
        walk_length: int,
        num_walks: int,
        output_directory: str,
        input_directory: Optional[str] = None,
) -> None:
    """Run the Node2Vec pipeline."""
    (node_path, edge_path, feature_path,
     validate_path, symptomatic_path, permutation_paths) = ensure_data(directory=input_directory)

    transition_probability_path = os.path.join(output_directory, 'transition_probabilities.json')

    graph = create_himmelstein_graph(node_path, edge_path)
    model = fit_node2vec(
        graph,
        transition_probabilities_path=transition_probability_path,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
    )

    embedder_function = EMBEDDERS[embedder]
    train_list, train_labels = train_pairs(feature_path)
    #  TODO why build multiple embedders separately and not single one then split vectors after the fact?
    train_vectors = embedder_function(model, train_list)
    disease_modifying, clinical_trials, drug_central, symptomatic = test_pairs(validate_path, symptomatic_path)
    test_dm_vectors = embedder_function(model, disease_modifying[0])
    test_dm_labels = disease_modifying[1]
    test_ct_vectors = embedder_function(model, clinical_trials[0])
    test_ct_labels = clinical_trials[1]
    test_dc_vectors = embedder_function(model, drug_central[0])
    test_dc_labels = drug_central[1]
    test_sy_vectors = embedder_function(model, symptomatic[0])
    test_sy_labels = symptomatic[1]

    _train_evaluate_generate_artifacts(output_directory, train_vectors, train_labels, test_dm_vectors, test_dm_labels,
                                       test_ct_vectors, test_ct_labels, test_dc_vectors, test_dc_labels,
                                       test_sy_vectors, test_sy_labels)


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


def _train_evaluate_generate_artifacts(
        output_directory,
        train_vectors,
        train_labels,
        test_dm_vectors,
        test_dm_labels,
        test_ct_vectors=None,
        test_ct_labels=None,
        test_dc_vectors=None,
        test_dc_labels=None,
        test_sy_vectors=None,
        test_sy_labels=None

) -> None:
    if not test_ct_vectors:
        test_dict = {'Test vectors': test_dm_vectors,
                     'Test labels': test_dm_labels}


    else:
        test_dict = {'Disease Modifying vectors': test_dm_vectors,
                     'Disease Modifying labels': test_dm_labels,
                     'Clinical Trial vectors': test_ct_vectors,
                     'Clinical Trial labels': test_ct_labels,
                     'Drug Central vectors': test_dc_vectors,
                     'Drug Central labels': test_dc_labels,
                     'Symptomatic vectors': test_sy_vectors,
                     'Symptomatic labels': test_sy_labels}
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
        json.dump(test_dict,
                  file,
                  indent=2,
                  sort_keys=True,
                  )

    click.echo('training logistic regression classifier')
    logistic_regression = train_logistic_regression(train_vectors, train_labels)
    with open(os.path.join(output_directory, 'logistic_regression_clf.joblib'), 'wb') as file:
        joblib.dump(logistic_regression, file)

    click.echo('validating logistic regression classifier')
    if not test_ct_vectors:

        roc, y_pro = validate(logistic_regression, test_dm_vectors, test_dm_labels)
        y_pro = list(map(list, y_pro))
        roc_dict = {'ROC:': roc,
                    'y_probability': y_pro}
    else:
        dm_roc, dm_yp = validate(logistic_regression, test_dm_vectors, test_dm_labels)
        ct_roc, ct_yp = validate(logistic_regression, test_ct_vectors, test_ct_labels)
        dc_roc, dc_yp = validate(logistic_regression, test_dc_vectors, test_dc_labels)
        sy_roc, sy_yp = validate(logistic_regression, test_sy_vectors, test_sy_labels)
        dm_yp = list(map(list, dm_yp))
        ct_yp = list(map(list, ct_yp))
        dc_yp = list(map(list, dc_yp))
        sy_yp = list(map(list, sy_yp))

        roc_dict = {

            "Disease Modifying": {
                "ROC": dm_roc,
                'Prediction Probability': dm_yp
            },

            'Clinical Trial': {
                'ROC': ct_roc,
                'Prediction Probability': ct_yp
            },
            'Drug Central': {
                'ROC': dc_roc,
                'Prediction Probability': dc_yp
            },
            'Syptomatic': {
                'ROC': sy_roc,
                'Prediction Probability': sy_yp
            },

        }
    with open(os.path.join(output_directory, 'validation.json'), 'w') as file:
        json.dump(
            roc_dict,
            file,

        )
