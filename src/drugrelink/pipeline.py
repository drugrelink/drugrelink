# -*- coding: utf-8 -*-

"""Main functions for running the repositioning comparison."""

import itertools as itt
import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

import click
import joblib
import numpy as np
import pandas as pd
from edge2vec import calculate_edge_transition_matrix, read_graph, train
from glmnet.logistic import LogitNet

from .constants import RESOURCES_DIRECTORY, NOTEBOOK_DIRECTORY
from .create_graph import create_himmelstein_graph
from .download import get_data_paths
from .embedders import get_embedder
from .graph_edge2vec import prepare_edge2vec
from .node2vec_utils import fit_node2vec
from .pairs import train_test_pairs, data_non_overlap, pairs_vectors
from .permutation_convert import convert
from .subgraph import generate_subgraph
from .train import train_logistic_regression, validate

logger = logging.getLogger(__name__)


def run_node2vec_graph(
        *,
        dimensions: int,
        walk_length: int,
        num_walks: int,
        window: int,
        embedder: str = "hadamard",
        permutation_number=None,
        output_directory: Optional[str] = None,
        input_directory: Optional[str] = None,
        repeat: int = 1,
        p: Optional[int] = None,
        q: Optional[int] = None,
) -> None:
    """
    Run the node2vec pipeline
    :param dimensions: The number of dimensions of embedding vectors
    :param walk_length: The length of one random walk.
    :param num_walks: The number of random walk iterating the graph
    :param window: Window size in word2vec
    :param embedder: Method of calculating edge vectors from node vectors
    :param permutation_number: The number of permutation graph, should be 1-5.
    :param output_directory: The path of output directory
    :param input_directory: The path of input directory
    :param repeat: Repeat times of one experiment
    :param p: Return hyper parameter
    :param q: Inout parameter
    :return: None
    """
    if output_directory is None:
        output_directory = os.path.join(RESOURCES_DIRECTORY,
                                        datetime.now().strftime(f'node2vec_%Y%m%d_%H%M'))
        os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, 'metadata.json'), 'w') as file:
        json.dump(
            {
                'dimensions': dimensions,
                'walk_length': walk_length,
                'num_walks': num_walks,
                'embedder': embedder,
                'input_directory': input_directory,
                'output_directory': output_directory,
                'window': window,
                'p': p,
                'q': q,
                'repeat': repeat,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    # validation_directory = os.path.join(output_directory,'validations')
    # os.makedirs(validation_directory, exist_ok=True)

    data_paths = get_data_paths(directory=input_directory)
    dir_number = 0
    for name in os.listdir(output_directory):
        path = os.path.join(output_directory, name)
        if os.path.isdir(path):
            dir_number += 1
    for i in range(dir_number + 1, repeat + 1):
        transition_probability_path = os.path.join(output_directory, 'transition_probabilities.json')

        sub_output_directory = os.path.join(output_directory, str(i))
        os.makedirs(sub_output_directory)
        if not permutation_number:
            graph = create_himmelstein_graph(data_paths.node_data_path, data_paths.edge_data_path)
        else:
            graph = convert(data_paths.permutation_paths[permutation_number - 1], permutation_number)

        model = fit_node2vec(
            graph,
            transition_probabilities_path=transition_probability_path,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            window=window,
            p=p,
            q=q,
        )
        model.save(os.path.join(sub_output_directory, 'word2vec_model.pickle'))
        embedder_function = get_embedder(embedder)
        #  TODO why build multiple embedders separately and not single one then split vectors after the fact?
        disease_modifying_training, disease_modifying, clinical_trials, drug_central, symptomatic = train_test_pairs(
            validation_path=data_paths.validate_data_path,
            symptomatic_path=data_paths.symptomatic_data_path,
            train_path=data_paths.transformed_features_path,
        )
        train_vectors = embedder_function(model, disease_modifying_training[:, 0:2])
        train_labels = disease_modifying_training[:, 2].tolist()
        test_dm_vectors = embedder_function(model, disease_modifying[:, 0:2])
        test_dm_labels = disease_modifying[:, 2].tolist()
        test_ct_vectors = embedder_function(model, clinical_trials[:, 0:2])
        test_ct_labels = clinical_trials[:, 2].tolist()
        test_dc_vectors = embedder_function(model, drug_central[:, 0:2])
        test_dc_labels = drug_central[:, 2].tolist()
        test_sy_vectors = embedder_function(model, symptomatic[:, 0:2])
        test_sy_labels = symptomatic[:, 2].tolist()
        _train_evaluate_generate_artifacts(
            sub_output_directory,
            train_vectors,
            train_labels,
            test_dm_vectors,
            test_dm_labels,
            test_ct_vectors,
            test_ct_labels,
            test_dc_vectors,
            test_dc_labels,
            test_sy_vectors,
            test_sy_labels,
        )


def run_edge2vec_graph(
        *,
        dimensions: int,
        walk_length: int,
        num_walks: int,
        window: int,
        embedder: str = "hadamard",
        output_directory: Optional[str] = None,
        input_directory: Optional[str] = None,
        repeat: int = 1,
        p: Optional[int] = None,
        q: Optional[int] = None,
        directed: bool = False,
        e_step: int,
        em_iteration: int,
        max_count: int
) -> None:
    if output_directory is None:
        output_directory = os.path.join(RESOURCES_DIRECTORY,
                                        datetime.now().strftime(f'edge2vec_%Y%m%d_%H%M'))
        os.makedirs(output_directory, exist_ok=True)

    data_paths = get_data_paths(directory=input_directory)
    edge_path = data_paths.edge_data_path
    data_edge2vec_path = data_paths.data_edge2vec_path
    if not os.path.exists(data_edge2vec_path):
        prepare_edge2vec(edge_path, data_edge2vec_path)
    graph = read_graph(data_edge2vec_path)
    if repeat:
        data_paths = get_data_paths(directory=input_directory)
        dir_number = 0
        for name in os.listdir(output_directory):
            path = os.path.join(output_directory, name)
            if os.path.isdir(path):
                dir_number += 1
        for i in range(dir_number + 1, repeat + 1):
            transition_probabilities_path = os.path.join(output_directory, 'transition_probabilities')
            if transition_probabilities_path is not None and os.path.exists(transition_probabilities_path):
                with open(transition_probabilities_path, 'rb') as file:
                    transition_probabilities = np.load(file)
                logger.info(f'Loaded pre-computed probabilities from {transition_probabilities_path}')
            else:
                transition_probabilities = calculate_edge_transition_matrix(
                    graph=graph,
                    directed=directed,
                    e_step=e_step,
                    em_iteration=em_iteration,
                    walk_epochs=num_walks,
                    walk_length=walk_length,
                    p=p,
                    q=q,
                    walk_sample_size=max_count,
                )

            if transition_probabilities_path is not None:
                logger.info(f'Dumping pre-computed probabilities to {transition_probabilities_path}')
                np.save(transition_probabilities_path, transition_probabilities)

            sub_output_directory = os.path.join(output_directory, str(i))
            os.makedirs(sub_output_directory)

            model = train(
                transition_matrix=transition_probabilities,
                graph=graph,
                number_walks=num_walks,
                walk_length=walk_length,
                p=p,
                q=q,
                size=dimensions,
                window=window,
            )
            model.save(os.path.join(sub_output_directory, 'word2vec_model.pickle'))
            model.wv.save_word2vec_format(os.path.join(sub_output_directory, 'word2vec_wv'))
            embedder_function = get_embedder(embedder)
            disease_modifying_training, disease_modifying, clinical_trials, drug_central, symptomatic = train_test_pairs(
                validation_path=data_paths.validate_data_path,
                symptomatic_path=data_paths.symptomatic_data_path,
                train_path=data_paths.transformed_features_path,
            )
            train_vectors = embedder_function(model, disease_modifying_training[:, 0:2])
            train_labels = disease_modifying_training[:, 2].tolist()
            test_dm_vectors = embedder_function(model, disease_modifying[:, 0:2])
            test_dm_labels = disease_modifying[:, 2].tolist()
            test_ct_vectors = embedder_function(model, clinical_trials[:, 0:2])
            test_ct_labels = clinical_trials[:, 2].tolist()
            test_dc_vectors = embedder_function(model, drug_central[:, 0:2])
            test_dc_labels = drug_central[:, 2].tolist()
            test_sy_vectors = embedder_function(model, symptomatic[:, 0:2])
            test_sy_labels = symptomatic[:, 2].tolist()
            _train_evaluate_generate_artifacts(
                sub_output_directory,
                train_vectors,
                train_labels,
                test_dm_vectors,
                test_dm_labels,
                test_ct_vectors,
                test_ct_labels,
                test_dc_vectors,
                test_dc_labels,
                test_sy_vectors,
                test_sy_labels,
            )
        logger.info(datetime.now())


def run_node2vec_subgraph(
        *,
        dimensions: int,
        walk_length: int,
        num_walks: int,
        n_train_positive: int = 5,
        n_train_negative: int = 15,
        embedder: str = "hadamard",
        output_directory: Optional[str] = None,
        input_directory: Optional[str] = None,
) -> None:
    if output_directory is None:
        output_directory = os.path.join(RESOURCES_DIRECTORY,
                                        datetime.now().strftime(f'node2vec_%Y%m%d_%H%M'))
        os.makedirs(output_directory, exist_ok=True)

    # TODO re-write metadata export

    # Define some output paths
    data_paths = get_data_paths(directory=input_directory)
    transition_probabilities_path = os.path.join(output_directory, 'transition_probabilities.json')

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
        click.echo('loaded pickled subgraph info')

    else:
        click.echo('creating graph')
        graph = create_himmelstein_graph(data_paths.node_data_path, data_paths.edge_data_path)

        click.echo('creating sub-graph')
        (
            subgraph,
            positive_list,
            positive_labels,
            negative_list,
            negative_labels,
        ) = generate_subgraph(
            data_paths.transformed_features_path,
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
    model = fit_node2vec(
        subgraph,
        transition_probabilities_path=transition_probabilities_path,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
    )

    click.echo('saving word2vec')
    model.save(os.path.join(output_directory, 'word2vec_model.pickle'))

    click.echo('generating vectors')
    embedder_function = get_embedder(embedder)
    positive_vectors = embedder_function(model, positive_list)
    negative_vectors = embedder_function(model, negative_list)

    train_vectors = positive_vectors[:n_train_positive] + negative_vectors[:n_train_negative]
    train_labels = positive_labels[:n_train_positive] + negative_labels[:n_train_negative]
    test_vectors = positive_vectors[n_train_positive:] + negative_vectors[n_train_negative:]
    test_labels = positive_labels[n_train_positive:] + negative_labels[n_train_negative:]

    _train_evaluate_generate_artifacts(
        output_directory,
        train_vectors,
        train_labels,
        test_vectors,
        test_labels,
    )
    click.echo(datetime.now())


def _train_evaluate_generate_artifacts(
        output_directory,
        train_vectors,
        train_labels,
        test_dm_vectors,
        test_dm_labels,
        repurpose_vectors,
        repurpose_labels,
        repo_vectors,
        repo_labels,
        test_ct_vectors=None,
        test_ct_labels=None,
        test_dc_vectors=None,
        test_dc_labels=None,
        test_sy_vectors=None,
        test_sy_labels=None,
) -> None:
    if not test_ct_vectors:
        test_dict = {
            'Test vectors': test_dm_vectors,
            'Test labels': test_dm_labels,
        }

    else:
        test_dict = {
            'Disease Modifying vectors': test_dm_vectors,
            'Disease Modifying labels': test_dm_labels,
            'Clinical Trial vectors': test_ct_vectors,
            'Clinical Trial labels': test_ct_labels,
            'Drug Central vectors': test_dc_vectors,
            'Drug Central labels': test_dc_labels,
            'Symptomatic vectors': test_sy_vectors,
            'Symptomatic labels': test_sy_labels,
        }
    with open(os.path.join(output_directory, 'train.json'), 'w') as file:
        json.dump(
            [
                dict(label=train_label, vector=train_vector)
                for train_vector, train_label in zip(train_vectors, train_labels)
            ],
            file,
        )

    with open(os.path.join(output_directory, 'test.json'), 'w') as file:
        json.dump(test_dict, file)

    logger.info('training logistic regression classifier')
    logit_net: LogitNet = train_logistic_regression(train_vectors, train_labels)
    with open(os.path.join(output_directory, 'logistic_regression_clf.joblib'), 'wb') as file:
        joblib.dump(logit_net, file)

    logger.info('validating logistic regression classifier')
    if not test_ct_vectors:
        roc_test, y_pro_test, y_labels_test, aupr_test = validate(logit_net, test_dm_vectors, test_dm_labels)
        roc_repurpose, y_pro_repurpose, y_labels_repurpose, aupr_repurpose = validate(logit_net, repurpose_vectors,
                                                                                      repurpose_labels)
        roc_repo, y_pro_repo, y_labels_repo, aupr_repo = validate(logit_net, repo_vectors, repo_labels)
        y_pro = list(map(list, y_pro_test))
        roc_dict = {
            'test_data': {
                'AUCROC:': roc_test,
                'y_probability': y_pro,
                'y_labels': y_labels_test,
                'AUPR': aupr_test},

            'repurposeDB': {
                'AUCROC': roc_repurpose,
                'y_probability': y_pro_repurpose,
                'y_labels': y_labels_repurpose,
                'AUPR': aupr_repurpose
            },

            'repoDB': {
                'AUCROC': roc_test,
                'y_probability': y_pro_repo,
                'y_labels': y_labels_repo,
                'AUPR': aupr_repo
            }
        }
    else:
        dm_roc, dm_yp, dm_pre = validate(logit_net, test_dm_vectors, test_dm_labels)
        ct_roc, ct_yp, ct_pre = validate(logit_net, test_ct_vectors, test_ct_labels)
        dc_roc, dc_yp, dc_pre = validate(logit_net, test_dc_vectors, test_dc_labels)
        sy_roc, sy_yp, sy_pre = validate(logit_net, test_sy_vectors, test_sy_labels)
        dm_yp = list(map(list, dm_yp))
        ct_yp = list(map(list, ct_yp))
        dc_yp = list(map(list, dc_yp))
        sy_yp = list(map(list, sy_yp))
        dm_pre = list(map(int, dm_pre))
        ct_pre = list(map(int, ct_pre))
        dc_pre = list(map(int, dc_pre))
        sy_pre = list(map(int, sy_pre))

        roc_dict = {
            "Disease Modifying": {
                "ROC": dm_roc,
                'Prediction Probability': dm_yp,
                'Predicted Label': dm_pre,
            },
            'Clinical Trial': {
                'ROC': ct_roc,
                'Prediction Probability': ct_yp,
                'Predicted Label': ct_pre,
            },
            'Drug Central': {
                'ROC': dc_roc,
                'Prediction Probability': dc_yp,
                'Predicted Label': dc_pre,
            },
            'Symptomatic': {
                'ROC': sy_roc,
                'Prediction Probability': sy_yp,
                'Predicted Label': sy_pre,
            },
        }

    with open(os.path.join(output_directory, 'validation.json'), 'w') as file:
        json.dump(roc_dict, file)
    click.echo('Misson Completed')


def run_edge2vec_subgraph(
        *,
        dimensions: int,
        walk_length: int,
        num_walks: int,
        window: int,
        embedder: str = "hadamard",
        output_directory: Optional[str] = None,
        input_directory: Optional[str] = None,
        p: Optional[int] = None,
        q: Optional[int] = None,
        directed: bool = False,
        e_step: int,
        em_iteration: int,
        max_count: int,
        n_train_positive: int = 5,
        n_train_negative: int = 15,

) -> None:
    if output_directory is None:
        output_directory = os.path.join(RESOURCES_DIRECTORY,
                                        datetime.now().strftime(f'edge2vec_%Y%m%d_%H%M'))
        os.makedirs(output_directory, exist_ok=True)

    # TODO re-write metadata export

    # Define some output paths
    data_paths = get_data_paths(directory=input_directory)
    edge_path = data_paths.edge_data_path
    data_edge2vec_path = data_paths.data_edge2vec_path

    if not os.path.exists(data_edge2vec_path):
        prepare_edge2vec(edge_path, data_edge2vec_path)

    transition_probabilities_path = os.path.join(output_directory, 'transition_probabilities.json')

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
        graph = read_graph(data_edge2vec_path)

        click.echo('creating sub-graph')
        (
            subgraph,
            positive_list,
            positive_labels,
            negative_list,
            negative_labels,
        ) = generate_subgraph(
            data_paths.transformed_features_path,
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

    click.echo('fitting edge2vec/word2vec')
    if transition_probabilities_path is not None and os.path.exists(transition_probabilities_path):
        with open(transition_probabilities_path, 'rb') as file:
            transition_probabilities = pickle.load(file)
        logger.info(f'Loaded pre-computed probabilities from {transition_probabilities_path}')
    else:
        transition_probabilities = calculate_edge_transition_matrix(
            graph=subgraph,
            directed=directed,
            e_step=e_step,
            em_iteration=em_iteration,
            walk_epochs=num_walks,
            walk_length=walk_length,
            p=p,
            q=q,
            walk_sample_size=max_count,
        )

    if transition_probabilities_path is not None:
        logger.info(f'Dumping pre-computed probabilities to {transition_probabilities_path}')
        with open(transition_probabilities_path, 'wb') as file:
            pickle.dump(transition_probabilities, file)

    model = train(
        transition_matrix=transition_probabilities,
        graph=subgraph,
        number_walks=num_walks,
        walk_length=walk_length,
        p=p,
        q=q,
        size=dimensions,
        window=window,
    )

    click.echo('saving word2vec')
    model.save(os.path.join(output_directory, 'word2vec_model.pickle'))

    click.echo('generating vectors')
    embedder_function = get_embedder(embedder)
    positive_vectors = embedder_function(model, positive_list)
    negative_vectors = embedder_function(model, negative_list)

    train_vectors = positive_vectors[:n_train_positive] + negative_vectors[:n_train_negative]
    train_labels = positive_labels[:n_train_positive] + negative_labels[:n_train_negative]
    test_vectors = positive_vectors[n_train_positive:] + negative_vectors[n_train_negative:]
    test_labels = positive_labels[n_train_positive:] + negative_labels[n_train_negative:]

    _train_evaluate_generate_artifacts(
        output_directory,
        train_vectors,
        train_labels,
        test_vectors,
        test_labels,
    )


def retrain_all(
        *,
        method: str,
        input_directory: str = None,
        output_directory: str = None,
        n_retrains: int = 10,
) -> List[str]:
    data_paths = get_data_paths(directory=input_directory)
    dataset = data_non_overlap(
        validation_path=data_paths.validate_data_path,
        symptomatic_path=data_paths.symptomatic_data_path,
        train_path=data_paths.transformed_features_path,
    )
    pdata = dataset.loc[dataset['label'] == 1]
    ndata = dataset.loc[dataset['label'] == 0].sample(n=len(pdata))
    train_data = pd.concat([pdata, ndata])
    train_labels = train_data[['label']]
    logit_net_paths = []
    for i in range(n_retrains):
        model_path = os.path.join(
            RESOURCES_DIRECTORY, 'predictive_model', method, str(i), 'word2vec_model.pickle',
        )
        if not output_directory:
            logit_net_path = os.path.join(
                RESOURCES_DIRECTORY, 'predictive_model', method, str(i), 'logistic_regression.joblib',
            )
        else:
            logit_net_path = os.path.join(output_directory, method, str(i), 'logistic_regression.joblib')

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        train_vectors = pairs_vectors(train_data, model)
        logit_net: LogitNet = train_logistic_regression(train_vectors, train_labels)
        with open(logit_net_path, 'wb') as file:
            joblib.dump(logit_net, file)
        logit_net_paths.append(logit_net_path)

    return logit_net_paths


def retrain(
        *,
        output_directory: str,
        input_directory: str = None
):
    print(NOTEBOOK_DIRECTORY)
    with open(os.path.join(NOTEBOOK_DIRECTORY, 'repurpose_overlap.json'), 'r') as file:
        repurpose = json.load(file)
    print('yes')
    data_paths = get_data_paths(directory=input_directory)
    disease_modifying_training, disease_modifying, clinical_trials, drug_central, symptomatic = train_test_pairs(
        validation_path=data_paths.validate_data_path,
        symptomatic_path=data_paths.symptomatic_data_path,
        train_path=data_paths.transformed_features_path,
    )

    repo = pd.read_csv(os.path.join(NOTEBOOK_DIRECTORY, 'repo_data.csv'), index_col=False)
    for i, name in enumerate(os.listdir(output_directory), start=1):

        if name in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            subpath = os.path.join(output_directory, name)

            embedder_function = get_embedder('hadamard')

            wv_path = os.path.join(subpath, 'word2vec_model.pickle')
            with open(wv_path, 'rb') as f:
                wv = pickle.load(f)
            model = wv
            train_vectors = embedder_function(model, disease_modifying_training[:, 0:2])
            train_labels = disease_modifying_training[:, 2].tolist()

            test_data = disease_modifying.tolist() + clinical_trials.tolist() + drug_central.tolist() + symptomatic.tolist()
            df_test = pd.DataFrame(test_data, columns=['drug', 'disease', 'status'])
            test_pos = df_test.loc[df_test['status'] == 1].to_numpy()
            test_vectors = embedder_function(model, test_pos[:, 0:2])
            test_labels = test_pos[:, 2].tolist()

            repurpose_vectors = np.array(embedder_function(wv, repurpose))
            repurpose_labels = np.array([1] * len(repurpose))
            repo_vectors = embedder_function(wv, repo.to_numpy()[:, 0:2])
            repo_labels = repo.to_numpy()[:, 2]
            _train_evaluate_generate_artifacts(
                subpath,
                train_vectors,
                train_labels,
                test_vectors,
                test_labels,
                repurpose_vectors,
                repurpose_labels,
                repo_vectors,
                repo_labels
            )

    logger.info(datetime.now())


def predict(
        *,
        method: str,
        compound_ids: List[str],
        disease_ids: List[str],
        lg_path_list: List[str],
        output_directory: str,
        n_models: int = 10,
):
    embedder_function = get_embedder('hadamard')

    results = defaultdict(list)
    for i in range(n_models):
        model_path = os.path.join(RESOURCES_DIRECTORY, 'predictive_model', method, str(i), 'word2vec_model.pickle')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        for disease_id, compound_id in itt.product(disease_ids, compound_ids):
            dc = [(f'Disease::{disease_id}', f'Compound::{compound_id}')]
            edge_embedding = embedder_function(model, dc)
            logistic_regression = joblib.load(lg_path_list[i])
            pre = logistic_regression.preict(edge_embedding)
            results[f'Disease::{disease_id}, Compound::{compound_id}'].append(pre)

    with open(os.path.join(output_directory, 'results.json')) as file:
        json.dump(results, file)

    return dict(results)
