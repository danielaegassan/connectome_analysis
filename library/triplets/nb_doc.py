#!/usr/bin/env python3

# %% [markdown]
"""This is a document in code explaining the usage of common-neighbors and other tools
for analyzing triplets in a graph.
"""

# %% [markdown]
"""
# Common Neighbors

We want to compute if subpopulations in a directed network show any bias towards having
common neighbors among source and target pairs.

We have implemented tools in the file `common_neighbors.py` that we will explore here.

The guideline we have followed is that this code cannot talk the language of brain circuits.
So we don't talk of cells, synapses, or synaptic directions.
To see how the computation works, we will need an adjacency matrix,
that we obtain in the next section from the NMC portal data we have cached.
This data can be replaced by any adjacency matrix and node-properties for rest of the notebook.
"""

# %% [markdown]
"""
## NMC-portal central-columns

We illustrate common neighbor biases with SSCX-adjacencies we have cached for the NMC-portal.
We will use custom packages that should be available in the virtual-environment used to load this
notebook. Otherwise we can move onto the main discussion about the tool.
"""

# %%
from pathlib import Path
from IPython.display import display
import logging
import pandas as pd
import numpy as np
from bluepy import Circuit
from nmc_portal_resources_cli.utils import helper
from factology.v2 import laboratory


LOG = logging.getLogger("Connectome Analysis")
LOG.setLevel(logging.DEBUG)

LOG.info("Explore common neighbor biases")


proj83 = Path("/gpfs/bbp.cscs.ch/project/proj83")
path_analyses_config = proj83 / "home" / "sood" / "portal" / "factology-v2" / "analyses" / "config.yaml"

lab = laboratory.Laboratory(path_analyses_config)
lab.set_workspace(Path.cwd())

circuit = Circuit(lab.config["measurements"]["CircuitConfig"])

circuit.connections = helper.load_connections(circuit, lab.config["measurements"]["analyses"])


# %% [markdown]
"""
The connections loaded above will contain a `pandas.DataFrame` for each of the 8 regional central-columns
in SSCx.
Next we need to obtain adjancency matrices and node properties for the dataframe of connections,
while anonymizing the node indices. Once again we use our packages.
"""
# %%
target_region = "S1DZO_Column"
cells = lab.helper.get_cells(circuit, "central_columns", target_region)
conns = lab.helper.get_connections(circuit, "central_columns", target_region)

edges, nodes = lab.helper.anonymize_gids(conns, cells)
adjacency = lab.get_adjacency(edges, nodes)

# %% [markdown]
"""
## Redine node `mtype`
Since we are working with a brain circuit, let us adapt the target-region data we have for
the common neighbor biase computation.
In the circuit, we qualified a cell's `mtype` to the layer in which it is found.
For some connectivity analyses we would like to study the effect of the shape of the cell's morphology,
which should be the same across all layer morphologies of the same class such as `TPC`s, or
the several interneuron types. We can group the circuit's `mtypes`, or just redefine them!
"""
def tag_morphology(lmtype):
    """..."""
    return '_'.join(lmtype.split('_'))

node_properties = nodes.assign(lmtype=nodes.mtype, mtype=nodes.mtype.apply(tag_morphology))

# %% [markdown]
"""
If the above seems obscure, we wouldn't be surprized. These are excerpts from a pipeline workflow
we are developing to produce the backend data for the BBP-Portals.
With the adjacency and node-properties loaded, we can move on to obtaining the common-neighbor biases.
"""

# %% [markdown]
"""
## Counting the number of common neighbors.

Let us use a symbology suitable for directed networks.
We use a graph object to encapsulate the adjacency and node-properties loaded above.
Thus the graph $G$ represents it's adjacency as a compressed CSR matrix.
We are interested in common neighbors in source and target subpopulations among the nodes.
Let us call the set of sources $S$ and that of targets $T$, and individual nodes that are
their members $s$ and $t$ respectively.
Thus we are interested in $C_{s, t}$, the number of common neighbors.
Given the directedness of the edges, there are four different kinds of common neighbors.
These are S --> X --> T, S --> X <-- T, S <-- X --> T, and S <-- X <-- T.

Of these we will be interested in two:

The first type of common neighbors of interest is $S --> X <-- T$, where the common neighbors
are `targets` of both $S$ and $T$. These can counted as a matrix multiplication of the
adjacency matrix and its transpose, subsetted to $S$ along the rows and $T$ along the columns.

\begin{equation}
C_{s, t} = \sum_{x} M_{s, x} * W_{x, t}
\end{equation}

where $M$ is the connectivity-matrix (\it{i.e.} `adjacency` in the Python code above.),
and $W$ is it's transpose. In vector notation,

\begin{equation}
\mathbf{C}[S, P] = \mathbf{M}[S, :] \times \mathbf{W} [:, T]
\end{equation}

The second type is $S <-- X --> T$, where the common neighbors are `sources` of both $S$ and $T$.
These will form the common neighbors that 'sources` to nodes in both $S$ and $T$.

In code,
"""
# %%

from connectome_analysis.library.triplets import common_neighbors

cn_analysis = common_neighbors.CommonNeighborAnalysis({})

graph = cn_analysis.Graph(adjacency, node_properties)

# %% [markdown]
"""
We compute common-neighbors by specifying a sub-population of sources, and targets.
These sub-populations are represented as edge-types:
"""
# %%

edge_types = graph.read_edge_types(described=(["layer", "mtype"], ["layer", "mtype"]))

display(edge_types)

# %% [markdown]
"""
For each of the edge-types, we can obtain sources and targets, and count the number of
common neighbor sources / targets,
"""

# %%
E = edge_types.iloc[0]
display(E)

of_sources, and_targets = graph.pair_nodes(as_described=E, mask=True)
count_sources = graph.count_common_sources_to(subpopulation=(of_sources, and_targets))

# %% [markdown]
"""
In addition to the count of common sources, we will also need the connectivity subset to
the subpopulation sources and targets,
"""

# %%
subadj = graph.subset_adjacency_of(subpopulation=(of_sources, and_targets))

# %% [markdown]
"""
which can be combined with the count of common-sources to obtain the bias of
connected source-target pairs over those not-connected in the number of their common neighbor sources.
"""
cn_sources_bias = cn_analysis.model_bias(common_neighbors=count_sources, given_connectivity=subadj)
display(cn_sources_bias)

# %% [markdown]
"""
While this was a demonstration of the bias for a single edge-type population in the graph,
we can compute biases for all the edge-types,
"""

G = graph
P = (["layer", "mtype"], ["layer", "mtype"])

LOG.info("Compute common neighbor sources bias")
cn_sources_bias = cn_analysis.evaluate_bias(common_neighbor="sources", in_graph=G, by_subpopulation=P)

LOG.info("Compute common neighbor targets bias")
cn_targets_bias = cn_analysis.evaluate_bias(common_neighbor="targets", in_graph=G, by_subpopulation=P)

cn_biases = pd.concat([cn_sources_bias, cn_targets_bias], axis=1, keys=["cn_sources", "cn_targets"])
