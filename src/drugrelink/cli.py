# -*- coding: utf-8 -*-

"""Command line interface for repositioning comparison."""

import json
import logging
import sys

import click

from .pipeline import run_edge2vec_graph, run_edge2vec_subgraph, run_node2vec_graph, run_node2vec_subgraph

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)


@click.command()
@click.argument('config', type=click.File())
@click.option('-v', '--debug', is_flag=True)
def main(config: str, debug: bool):
    """This cli runs the ComparisonNRL."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Interpret as JSON file
    config = json.load(config)
    # Get high level configuration
    method = config['method'] # by default, use node2vec
    graph_type = config.pop('graph_type', 'graph')  # by default, do the whole graph
    # Choose the appropriate function then pass the rest of the configuration there
    # using the splat operator
    if not config['retrain'] and config['predict']:

        if method == 'node2vec':
            if graph_type == 'graph':
                return run_node2vec_graph(**config)

            elif graph_type == 'subgraph':
                return run_node2vec_subgraph(**config)

            elif graph_type == 'permutation':
                return run_node2vec_graph(**config)

            else:
                click.echo(f'Unsupported graph_type={graph_type}')
                return sys.exit(1)



        elif method == 'edge2vec':
            if graph_type == 'graph':
                return run_edge2vec_graph(**config)
            elif graph_type == 'subgraph':
                return run_edge2vec_subgraph(**config)

            else:
                click.echo(f'Unsupported graph_type={graph_type}')
                return sys.exit(1)

        else:
            click.echo(f'Unsupported method={method}')
            return sys.exit(1)

    elif config['retrain']:
       return retrain(**config)

    else config['predict']:



if __name__ == '__main__':
    main()
