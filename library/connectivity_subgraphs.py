#!/usr/bin/env python3

"""Methods to subset connectivity matrices.
"""
from collections import OrderedDict
from itertools import product

import logging
import numpy as np
import pandas as pd
from scipy import sparse

from bluepy import Cell, Synapse

LOG = logging.getLogger("connectome-analysis-topology")


def subgraph_by_layers(adjacency, node_properties, **kwargs):
    """Subset an adjacency matrix by the nodes's layers.
    """
    assert "layer" in node_properties, ("Missing layer among node_properties columns: "
                                        f"{node_properties.columns}")

    adj = adjacency.toarray()
    by_layer = node_properties.reset_index().groupby("layer").apply(lambda g: g["index"].values)
    subgraph_adjacencies = by_layer.apply(lambda nodes: sparse.csr_matrix(adj[nodes, :][:, nodes]))
    and_nodes = by_layer.apply(lambda nodes: node_properties.reindex(nodes).reset_index(drop=True))

    return pd.concat([subgraph_adjacencies, and_nodes], axis=1, keys=["adjacency", "node_properties"])


def specify_edges(by):
    """..."""
    if isinstance(by, str):
        edge_specifiers = (by, by)
    else:
        assert isinstance(by, tuple) and len(by) == 2, f"Invalid edge type by {by}"
        edge_specifiers = by
    return edge_specifiers


def get_node_types(node_properties, by, label):
    """..."""
    if not isinstance(by, list):
        by = [by]
    node_types = node_properties[by]
    return node_types.drop_duplicates().reset_index(drop=True).rename(columns=(label+"_{}").format)


def get_edge_types(node_properties, by):
    source_spec, target_spec = specify_edges(by)

    source_types = get_node_types(node_properties, source_spec, "source")
    target_types = get_node_types(node_properties, target_spec, "target")

    edge_types = source_types.join(target_types, how="cross")

    def multindex(column):
        """..."""
        split = column.split('_')
        return (split[0], '_'.join(split[1:]))

    edge_types.columns = pd.MultiIndex.from_tuples([multindex(c) for c in edge_types.columns])
    return edge_types


def extract_edge_type(adj, nodes):
    """..."""
    def get_nodes(of_type):
        """..."""
        selection = nodes[of_type.index.values] == of_type.values
        return np.all(selection, axis=1)

    def method(E):
        sources = get_nodes(E.source)
        targets = get_nodes(E.target)

        either = np.logical_or(sources, targets)
        subnodes = nodes[either].reset_index(drop=True)

        index = -1 * np.ones_like(either)
        index[either] = np.arange(either.sum())
        subadj = adj[either, :][:, either]

        only_sources = np.logical_and(sources, ~targets).astype(bool)
        subadj[:, index[only_sources]] = 0

        only_targets = np.logical_and(~sources, targets).astype(bool)
        subadj[index[only_targets], :] = 0

        return pd.Series({"adjacency": subadj, "node_properties": subnodes})

    return method


def subgraph(adjacency, nodes, by, **kwargs):
    """..."""
    LOG.info("Subgraph a graph with %s nodes by %s", adjacency.shape[0], by)
    edge_types = get_edge_types(nodes, by)
    LOG.info("..Producing a subgraph for each of %s edge types", len(edge_types))

    eindex = pd.MultiIndex.from_frame(edge_types, names=['_'.join(c) for c in edge_types])
    return edge_types.apply(extract_edge_type(adjacency.toarray(), nodes), axis=1).set_index(eindex)
