#!/usr/bin/env python3

"""Analyses that consider functions of three nodes at a time, for example common neighbors.
"""
from collections.abc import Mapping
from lazy import lazy
import logging
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("connectome-analysis-topology")
LOG.setLevel(logging.DEBUG)

def read_yaml(filepath):
    """..."""
    with open(filepath, 'r') as from_file:
        config = yaml.load(from_file, Loader=yaml.FullLoader)
    return config


class Parameters:
    """Parameterize the computation of common-neighbors...
    """
    def __init__(self, config):
        """..."""
        self._config = config

    @lazy
    def config(self):
        """..."""
        if isinstance(self._config, Mapping):
            return self._config

        filepath = Path(self._config)

        return read_yaml(filepath)

    @lazy
    def sample_fraction(self):
        """Fraction of nodes to sample.
        """
        return self.config.get("sample_fraction", 1.)

    @lazy
    def cn_epsilon(self):
        """TODO: What is this suppsed to be?
        """
        return self.config.get("cn_epsilon", 1.e-4)

    @lazy
    def limit_cn_to(self):
        """TODO: What is this supposed to be?
        """
        return self.config.get("limit_cn_to", None)

    @lazy
    def resolution(self):
        """TODO: What is this resolution of ?
        """
        return self.config.get("distance_resolution", 50.0)

    @lazy
    def nbins(self):
        """TODO: What are these bins of?
        """
        return self.config.get("distance_bins", None)

    @lazy
    def result_type(self):
        """TODO: What is the meaning of these terms?
        TODO: Make result type enum
        """
        return self.config.get("result_type", "pathway-distance-dependence")

    @lazy
    def min_samples_for_good_statistics(self):
        """..."""
        return self.config.get("require-minimum-number-samples", 5)


def redefine_mtype(node_properties):
    """Redefine mtype in node properties.
    The arbors and dendrites of TPC:A is similar despite the layer they are specialized to.
    We will redefine mtype, removing the layer specified in the circuit's mtype definition.
    The circuit's definition will be renamed to `lmtype`.
    """
    circuit_mtypes = node_properties.mtype
    redefine_mtypes = ['_'.join(m.split('_')[1:]) for m in circuit_mtypes]
    return node_properties.assign(lmtype=circuit_mtypes, mtype=redefine_mtypes)


def mask_sample(node_properties, fraction):
    """..."""
    N = len(node_properties)
    to_sample = np.random.choice(N, int(N * fraction), replace=False)
    mask = np.zeros(len(node_properties)).astype(bool)
    mask[to_sample] = True
    return mask


class CommonNeighborAnalysis:
    """Analyze common neighbors.
    """
    def __init__(self, parameters):
        """..."""
        self._parameters = parameters

    @lazy
    def parameters(self):
        """..."""
        return Parameters(self._parameters)

    class Graph:
        """A graph will contain the adjacency matrix and node properties passed for computation
        to the methods in this `CommonNeighborAnalysis`.
        """
        def __init__(self, adjacency, node_properties):
            """..."""
            self._adjacency = adjacency
            self._node_properties = node_properties

        @lazy
        def adjacency(self):
            """We expect the input adjacency defining this `Graph` to be a compressed 2D matrix.
            To get the array itself, use `Graph.adjmat`.
            """
            return self._adjacency

        @lazy
        def transpose(self):
            """...The same thing, but the matrix is transposed.
            This assumes that `Graph.adjacency` can be transposed in its compressed format.
            """
            return CommonNeighborAnalysis.Graph(self.adjacency.transpose(), self.node_properties)

        @lazy
        def adjmat(self):
            """..."""
            return self._adjacency.toarray()

        @lazy
        def node_properties(self):
            """..."""
            return self._node_properties

        def specify_edges(self, described):
            """..."""
            if isinstance(described, str):
                edge_specifiers = (described, described)
            else:
                assert isinstance(described, tuple) and len(described) == 2,\
                    f"Invalid edge type {described}"
                edge_specifiers = described

            return edge_specifiers

        def read_types_of(self, node, described):
            """...Get the types of sources or targets.
            nodes: "source" or "target"
            in_description: a list of node-property variables  (i.e. columns)
            """
            assert node in ("source", "target")
            if not isinstance(described, list):
                described = [described]

            node_types = self.node_properties[described]
            return node_types.drop_duplicates().reset_index(drop=True).rename(columns=(node +"_{}").format)

        @staticmethod
        def pd_MultiIndex(columns):
            """..."""
            def multiindex(c):
                s = c.split('_')
                return (s[0], '_'.join(s[1:]))
            return pd.MultiIndex.from_tuples([multiindex(c) for c in columns])

        def read_edge_types(self, described):
            """Get edge types in graph by a description...
            """
            by_sources, by_targets = self.specify_edges(described)

            source_types = self.read_types_of("source", described=by_sources)
            target_types = self.read_types_of("target", described=by_targets)

            edge_types = source_types.join(target_types, how="cross")
            edge_types.columns = self.pd_MultiIndex(edge_types.columns)
            return edge_types

        def mask_nodes(self, as_described):
            """Get a boolean mask for nodes that have been described as node-property-value pairs,
            in a `Mapping` or  `pandas.Series`.

            We first get a `pandas.Series` for the nodes required `as_described` ---
            expecting that the series will contain an index naming the node-properties and contain their values.

            Example:
            Consider that we want `cn_bias` for each `mtype->mtype` pathway.
            We can define a `CommonNeighborAnalysis` instance that will do this.
            To then mask the nodes as described by a pathway's sources or targets,
            we can call
            ```
            common_neighbors.mask_nodes({"layer": "L2", "mtype": "MC"})
            ```
            Notice that in the above description we are using the redefined node-property of `mtype`

            TODO: Considering that this library of code should be independent of the research domain that
            the graphs originate in, we should move the redefinition of `mtype` to another package that should
            know what a brain-circuit network is.
            """
            of_type = pd.Series(as_described)
            return np.all(self._node_properties[of_type.index.values] == of_type.values, axis=1)

        def find_nodes(self, as_described, only_ids=False, as_mask=False):
            """Find node (or their ids) as described.

            TODO: Can we let subpopulation of nodes `as_described` be an array of node-ids, or
            as a boolean mask?
            Array of node-ids will need to use .loc...
            """
            def from_array():
                """Nodes from an array."""
                assert as_described.ndim == 1
                masked = self.node_properties.apply(lambda _: False, axis=1)
                masked[as_described] = True
                return masked

            def from_cell_type():
                """Nodes from a cell type description."""
                return self.mask_nodes(as_described)

            def nodes(masked):
                return self.node_properties[masked]

            def ids(masked):
                return self.node_properties.index[masked].values

            masked = from_array() if isinstance(as_described, np.ndarray) else from_cell_type()

            return masked if as_mask else (ids(masked) if only_ids else nodes(masked))

        def describe_pair(self, subpopulations):
            """We describe the (source, target) subpopulations as either
            1) a singleton of
            ~  a) 1D-array of node ids or
            ~  b) A description of node-types,
            ~  or
            2) a pair of
            ~  a) 1D-array of source ids, 1D-array of target ids
            ~  b) 1D-array of source ids, description of targets
            ~  c) description of sources, 1D-array of target ids
            ~  d) description of sources, description of targets
            """
            if isinstance(subpopulations, tuple):
                return subpopulations

            subpopulation = pd.Series(subpopulations)
            try:
                subsources = subpopulations.source
            except AttributeError:
                subsources = subtargets = subpopulations
            else:
                subtargets = subpopulations.target
            return (subsources, subtargets)

        def pair_nodes(self, as_described, mask=False):
            """Find the pairs of potential sources and targets in a sub-population as described
            """
            sources, targets = self.describe_pair(as_described)
            return ((self.find_nodes(sources, only_ids=True), self.find_nodes(targets, only_ids=True))
                    if not mask else
                    (self.find_nodes(sources, as_mask=True), self.find_nodes(targets, as_mask=True)))

        def count_common_sources_to(self, subpopulations):
            """Count the number of nodes in graph that are sources of edges to each of the nodes among
            pairs in a subpopulation described by a tuple of sources and targets.

            All pairs of sources and targets will be represented as a matrix...
            """
            subsources, subtargets = subpopulations
            edges_to_subsources = self.transpose.adjacency[subsources, :].astype(int)
            edges_to_subtargets = self.adjacency[:, subtargets].astype(int)
            return edges_to_subsources * edges_to_subtargets

        def count_common_targets_of(self, subpopulations):
            """Count the number of nodes in graph that are targets of edges from each of the nodes among
            pairs in a subpopulation described by a tuple of sources and targets.

            All pairs of sources and targets will be represented as a matrix...
            """
            subsources, subtargets = subpopulations
            edges_from_subsources = self.adjacency[subsources, :].astype(int)
            edges_from_subtargets = self.transpose.adjacency[:, subtargets].astype(int)
            return edges_from_subsources * edges_from_subtargets

        def subset_adjacency_of(self, subpopulations):
            """Subadjacency matrix of a subpopulation of nodes.
            """
            of_sources, and_targets = subpopulations
            return self.adjacency[of_sources, :][:, and_targets]

    def EdgeType(self, from_source_type, to_target_type):
        """..."""
        source_type = pd.Series(from_source_type)
        target_type = pd.Series(to_target_type)
        return pd.concat([source_type, target_type])

    def specify_subpopulations(self, described):
        """Get the edge-type described as a pathway.
        """
        if isinstance(described, (Mapping, pd.Series)):
            return self.EdgeType(from_source_type=described, to_target_type=described)

        if isinstance(described, tuple):
            assert len(tuple) == 2,\
                f"The tuple description {described} should describe (sources, targets)."

            return self.EdgeType(from_source_type=described[0], to_target_type=described[1])

        raise NotImplementedError(f"Specification of edge type for {type(described)}")

    def subgraph_adjacency(self, in_graph, of_subpopulations):
        """Extract the sub-matrix adjacency of a subpopulation of nodes in a graph.
        """
        of_sources, and_targets = in_graph.pair_nodes(of_subpopulations, mask=True)
        return in_graph.subset_adjacency_of(subpopulations=(of_sources, and_targets))

    def count(self, common_neighbor, in_graph, of_subpopulations):
        """...Count the number of common neighbors of a population of nodes
        among connections in a connectivity matrix provided in a `Graph` instance

        common_neighbor :: String #The type of common neighbors.
        in_graph: Whole population adjacency and node-properties in a `CommonNeighborAnalysis.Graph`.
        of_subpulation: of the nodes in the connectivity matrix to compute neighbors of
        """
        of_sources, and_targets = in_graph.pair_nodes(of_subpopulations, mask=True)

        if common_neighbor == "sources":
            return in_graph.count_common_sources_to(subpopulations=(of_sources, and_targets))

        if common_neighbor == "targets":
            return in_graph.count_common_targets_of(subpopulations=(of_sources, and_targets))

        raise ValueError(f"Unknown or unimplemented common-neighbor type %s", common_neighbor)

    def has_enough_statistics(self, connectivity):
        """An analysis described in code should also have facilities for statistical
        evaluation. For any statistical evaluation, we can check if a connectivity matrix
        has enough samples.
        """
        return connectivity.nnz >= self.parameters.min_samples_for_good_statistics

    def model_bias(self, common_neighbors, given_connectivity):
        """Evaluate bias in the count of common neighbors  given connectiviity of the nodes

        common_neighbors : np.ndarray dim 2, of number of common neighbors among nodes
        ~                  represented by the indices. Here we assume that the matrix is of shape S x P,
        ~                  where S is the number of source nodes in a sub-population
        ~                  and P the number of target nodes in a sub-population.
        given_connectivity: np.ndarray dim 2, of the same shape S x P, that gives boolean indicating
        ~                   if a source is connected to a target

        TODO: The method could use some meta-information stored in this `CommonNeighborAnalysis` instance.
        TODO: We do not compute the distance dependence in the model yet. So, implement distance dependence
        ~     as part of the `Parameters` used by `CommonNeighborAnalysis` instance.
        """
        def output(bias, pvalue=None, intercept=None, added=None):
            return pd.Series({"bias": bias, "pvalue": pvalue,
                              "model_intercept": intercept, "model_added": added})

        if not self.has_enough_statistics(given_connectivity):
            return output(None)

        from statsmodels.formula import api as StatModel
        from patsy import ModelDesc

        input_data = {"CN": common_neighbors.toarray().flatten(),
                      "Connected": given_connectivity.astype(bool).toarray().flatten()}

        description = ModelDesc.from_formula("CN ~ Connected")
        model = StatModel.ols(description, input_data).fit()

        pvalue = model.pvalues.get("Connected[T.True]", 1.0)
        added = model.params.get("Connected[T.True]", 0.0)
        intercept = model.params["Intercept"]
        bias = added / intercept
        return output(bias, pvalue, intercept, added)


    def extract_adj_and_counts(self, of_common_neighbor, in_graph, for_edge_type):
        """Obtain the submatrices representing the adjacency and the number of common neighbors,
        of sources to target pairs in given edge type.

        Arguments
        ------------------------------------------------------------------------------------------
        of_common_neighbor :: String #Either "sources", or "targets"

        in_graph :: Graph

        for_edge_type :: Either(Mapping or Tuple(Mapping, Mapping) #that specifies the edge type,
        ~                where `Mapping` is that of cell properties to their cell properties that
        ~                define the source and target subpopulations.
        ~ examples: ```
                for_edge_type={"layer": 1, "mtype": "DAC"}
                for_edge_type=tuple({"layer": 1, "mtype": "DAC"}, {"layer": 3, "mtype": "TPC:A"})
        ```
        ------------------------------------------------------------------------------------------
        NOTE: The populations of sources and targets in the argued edge type may not be identical,
        of the same number.
        TODO: Refactor `Graph` to allow for different populations along the rows, and columns.
        This will required that there be two node properties, one for sources, i.e. rows, and
        another for columns, i.e targets.
        """
        by_edge_type = self.specify_subpopulations(for_edge_type)
        counts = self.count(of_common_neighbor, in_graph, of_subpopulations=by_edge_type)
        matrix = self.subgraph_adjacency(in_graph, of_subpopulations=by_edge_type)

        return (counts, matrix)

    def evaluate_bias(self, of_common_neighbor, in_graph, for_edge_type):
        """Evaluate bias for connectivity in the number of common neighbors of
        sources and targets of a given edge-type.

        Arguments
        ------------------------------------------------------------------------------------------
        of_common_neighbor :: String #Either "sources", or "targets"

        in_graph :: Graph

        for_edge_type :: Either(Mapping or Tuple(Mapping, Mapping) #that specifies the edge type,
        ~                where `Mapping` is that of cell properties to their cell properties that
        ~                define the source and target subpopulations.
        ~ examples: ```
                for_edge_type={"layer": 1, "mtype": "DAC"}
                for_edge_type=tuple({"layer": 1, "mtype": "DAC"}, {"layer": 3, "mtype": "TPC:A"})
        ```
        ------------------------------------------------------------------------------------------
        """
        LOG.debug("Evaluate common neighbor bias for edge type \n%s", for_edge_type)
        counts, matrix = self.extract_adj_and_counts(of_common_neighbor, in_graph, for_edge_type)
        return self.model_bias(common_neighbors=counts, given_connectivity=matrix)

    def evaluate_sources_bias(self, in_graph, for_edge_type):
        """Evaluate the connectivity bias of common sources in a graph for a given edge type.
        """
        G = in_graph
        E = for_edge_type
        return self.evaluate_bias(of_common_neighbor="sources", in_graph=G, for_edge_type=E)

    def evaluate_targets_bias(self, in_graph, for_edge_type):
        """Evaluate the connectivity bias of common sources in a graph for a given edge type.
        """
        G = in_graph
        E = for_edge_type
        return self.evaluate_bias(of_common_neighbor="targets", in_graph=G, for_edge_type=E)

    def evaluate_bias_by_pathways(self, groupedby, of_common_neighbor, in_graph):
        """Evaluate connectivity bias in the number of common neighbors between sources and targets
        for each of such subpopulations in pathways that are grouped by specifying node-properties
        of the sources and targets.
        """
        edge_types = in_graph.read_edge_types(groupedby)
        by_edges = pd.MultiIndex.from_frame(edge_types, names=['_'.join(c) for c in edge_types])

        evaluate = lambda E: self.evaluate_bias(of_common_neighbor, in_graph, for_edge_type=E)
        return edge_types.apply(evaluate, axis=1).set_index(by_edges)

    def evaluate_sources_bias_by_pathways(self, groupedby, in_graph):
        """Evaluate connectivity bias in the number of common neighbor sources between
        sources and targets among grouped by either one or a list of node-properties.
        Node-properties to group sources and targets may be specified seprately in a two-tuple.

        Arguments
        --------------------------------------------------------------------------------------
        groupedby : Either[List[NodeProperty], Tuple[List[NodeProperty]]]
        --------------------------------------------------------------------------------------
        """
        G = in_graph
        return self.evaluate_bias_by_pathways(groupedby, of_common_neighbor="sources", in_graph=G)

    def evaluate_targets_bias_by_pathways(self, groupedby, in_graph):
        """Evaluate connectivity bias in the number of common neighbor targets between
        sources and targets among grouped by either one or a list of node-properties.
        Node-properties to group sources and targets may be specified seprately in a two-tuple.

        Arguments
        --------------------------------------------------------------------------------------
        groupedby : Either[List[NodeProperty], Tuple[List[NodeProperty]]]
        --------------------------------------------------------------------------------------
        """
        G = in_graph
        return self.evaluate_bias_by_pathways(groupedby, of_common_neighbor="targets", in_graph=G)



# The classes defined above will be used by the methods below that can be exposed as the library
# of common-neighbor methods
def get_common_neighbor_biases(adjacency, node_properties, *,
                               by_node_properties=None, for_edge_type=None, **parameters):
    """Compute all the possible cn-biases.

    Arguments
    ------------------------------------------------------------------------------------------------
    by_node_properties :: Either[List, Tuple(List, List)] #that provides node properties to group
    ~ sources, and targets to compute biases for.
    ~ examples
    ~  (["layer", "mtype"], ["layer", "mtype"])

    for_edge_type :: Either[Mapping, Tuple(Mapping, Mapping)] #that provides mappings of node-properties
    ~ to values to filter the sources and targets to copmute biases for.
    ~ examples
    ~  ({"layer": 1}, {"layer": 3, "mtype": "TPC:A"})

    TODO:
    parameters :: Mapping[CommonNeighborAnalysis.Parameters --> values] #that parameterizes the kind of
    ~ common-neighbor biases to compute

    ------------------------------------------------------------------------------------------------
    """
    assert not (by_node_properties is not None and for_edge_type is not None),\
        "Get common neighbor biases by either listing node properties or by mapping edge types, not both."

    cn_analysis = CommonNeighborAnalysis(parameters)
    of_adjacency = cn_analysis.Graph(adjacency, node_properties)
    layers_and_mtypes = (["layer", "mtype"], ["layer", "mtype"])

    def evaluate_bias_of_common(sources_or_targets):
        return cn_analysis.evaluate_bias_by_pathways(grouped_by=layers_and_mtypes,
                                                     of_common_neighbor=sources_or_targets,
                                                     in_graph=of_adjacency)

    cn_sources_bias = evaluate_bias_of_common("sources")
    cn_targets_bias = evaluate_bias_of_common("targets")

    return pd.concat([cn_sources_bias, cn_targets_bias], axis=1, keys=["cn_sources", "cn_targets"],
                     names=["cn_types"])
