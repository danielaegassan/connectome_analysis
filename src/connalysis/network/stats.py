# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Functions to average (functional or structural) metrics across simplices or neighborhoods.
Author(s): Daniela Egas Santander,
Last update: 11.2023
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import operator

from .local import neighborhood_indices


def node_stats_per_position_single(simplex_list, values, with_multiplicity=True):
    """ Get mean, standard deviation and standard error of the mean averaged across simplex lists and filtered per position
    Parameters
    ----------
    simplex_list : 2d-array
        Array of dimension (no. of simplices, dimension) listing simplices to be considered.
        Each row corresponds to a list of nodes on a simplex indexed by the order of the nodes in an NxN matrix.
        All entries must be an index in values
    values : Series
        pandas Series with index the nodes of the NxN matrix of which the simplices are listed,
        and values the values on that node to be averaged.
    with_multiplicity : bool
        if ``True`` the values are averaged with multiplicity i.e., they are weighted by the number of times a node
        participates in a simplex in a given position
        if ``False`` repetitions of a node in a given position are ignored.

    Returns
    -------
    DataFrame
        with index, the possible positions of o node in a ``k``-simplex and columns the mean, standard deviation and
        standard error of the mean for that position
    """
    # Filter values
    if with_multiplicity:
        vals_sl = values.loc[simplex_list.flatten()].to_numpy().reshape(simplex_list.shape)
    else:
        vals_sl = pd.concat([values.loc[np.unique(simplex_list[:, pos])] for pos in range(simplex_list.shape[1])],
                            axis=1, keys=range(simplex_list.shape[1]))
    # Compute stats
    stats_vals = pd.DataFrame(index=pd.Index(range(simplex_list.shape[1]), name="position"))
    # Stats per position
    stats_vals["mean"] = np.nanmean(vals_sl, axis=0)
    stats_vals["std"] = np.nanstd(vals_sl, axis=0)
    stats_vals["sem"] = stats.sem(vals_sl, axis=0, nan_policy="omit")
    # Stats in any position
    stats_vals.loc["all", "mean"] = np.nanmean(vals_sl)
    stats_vals.loc["all", "std"] = np.nanstd(vals_sl)
    stats_vals.loc["all", "sem"] = stats.sem(vals_sl, axis=None, nan_policy="omit")
    return stats_vals

def node_stats_per_position(simplex_lists, values, dims=None, with_multiplicity=True):
    """ Get across dimensions mean, standard deviation and standard error of the mean averaged across simplex lists
    and filtered per position
    Parameters
    ----------
    simplex_lists : dict
        keys : are int values representing dimensions
        values : for key ``k`` array of dimension (no. of simplices, ``k``) listing simplices to be considered.
        Each row corresponds to a list of nodes on a simplex indexed by the order of the nodes in an NxN matrix.
        All entries must be an index in values
    values : Series
        pandas Series with index the nodes of the NxN matrix of which the simplices are listed,
        and values the values on that node to be averaged.
    with_multiplicity : bool
        if ``True`` the values are averaged with multiplicity i.e., they are weighted by the number of times a node
        participates in a simplex in a given position
        if ``False`` repetitions of a node in a given position are ignored.
    dims : iterable
        dimensions for which to run the analysis, if ``None`` all the keys of simplex lists will be analyzed

    Returns
    -------
    dict
        keys the dimensions anlayzed and values for key ``k`` a DataFrame
        with index, the possible positions of o node in a ``k``-simplex and columns the mean, standard deviation and
        standard error of the mean for that position.
    """
    if dims is None:
        dims = simplex_lists.index
    stats_dict = {}
    for dim in tqdm(dims):
        sl = simplex_lists.loc[dim]
        stats_dict[dim] = node_stats_per_position_single(sl, values, with_multiplicity=with_multiplicity)
    return stats_dict


def node_stats_participation(participation, vals, condition=operator.eq, dims=None):
    """ Get statistics of the values in vals across nodes filtered using node participation
    Parameters
    ----------
    participation : DataFrame
        DataFrame of node participation with index the nodes in nodes of an NxN matrix to consider,
        columns are dimensions and values are node participation computed with
        [node_participation](network_topology.md#src.connalysis.network.topology.node_participation).
    values : Series
        pandas Series with index the nodes of the NxN matrix of where node participation has been computed
        and vals the values on that node to be averaged.
    condition : operator
        operator with which to filter the nodes. The default ``operator.eq`` filters nodes such that their maximal
        dimension of node participation is a given value.
        Alternatively, ``operator.ge`` filters nodes such that their maximal dimension of node participation is at least a given
        value.
    dims : iterable
        dimensions for which to run the analysis, if ``None`` all the columns of participation will be analyzed

    Returns
    -------
    DataFrame
        with index, the dimensions for which the analysis have been run and columns the statistics of the values in vals
        where the nodes have been grouped according to the condition given.

    See Also
    --------
    [node_stats_per_position_single](network_stats.md#src.connalysis.network.stats.node_stats_per_position_single):
    A similar function where the position of the nodes in the simplex are taken into account.  Note in particular that
    if condition = ``operator.ge`` the weighted_mean of this analyisis is equivalent than the value given by this function for position ``all``.
    However the computation using
    [node_participation](network_topology.md#src.connalysis.network.topology.node_participation)
    is more efficient.
    """
    par_df = participation.copy()
    if dims is None:
        dims = par_df.columns
    vals = vals.rename("values")
    par_df["max_dim"] = (par_df > 0).sum(axis=1) - 1  # maximal dimension a node is part of
    stats_vals = {}
    for dim in dims:
        mask = condition(par_df.max_dim, dim)
        c = pd.DataFrame(vals.loc[par_df[mask].index], columns=["values"])
        c["weight"] = par_df[mask][dim]
        w_mean = c.apply(np.product, axis=1).sum() / (c["weight"].sum())
        stats_vals[dim] = (c.shape[0],  # Number of nodes fulfilling the condition
                           np.nanmean(c["values"]),
                           np.nanstd(c["values"]),
                           stats.sem(c["values"], nan_policy="omit"),
                           w_mean  # mean weighted by participation
                           )
    stats_vals = pd.DataFrame.from_dict(stats_vals, orient="index",
                                        columns=["counts", "mean", "std", "sem", "weighted_mean"])
    stats_vals.index.name = "dim"
    return stats_vals


def node_stats_neighborhood(values, adj=None, pre=True, post=True, all_nodes=True, centers=None,
                            include_center=True, precomputed=False, neighborhoods=None):
    """ Get basic statistics of the property values on the neighbhood of the nodes in centers in the
    graph described by adj.
    Parameters
    ----------
    values : Series
        pandas Series with index the nodes of the NxN matrix of which the simplices are listed,
        and values the values on that node to be averaged.
    adj : sparse matrix or 2d array
        The adjacency matrix of the graph
    pre : bool
        If ``True`` compute the nodes mapping to the nodes in centers (the in-neighbors of the centers)
    post : bool
        If ``True`` compute the nodes that the centers map to (the out-neighbors of the centers)
    all_nodes : bool
        If ``True`` compute the neighbors of all nodes in adj, if ``False`` compute only the neighbors of the nodes
        listed in centers
    centers : 1d-array
        The indices of the nodes for which the neighbors need to be computed.  This entry is ignored if
        all_nodes is ``True`` and required if all_nodes is ``False``
    include_center : bool
        If ``True`` it includes the center in the computation otherwise it ignores it
    precomputed : bool
        If ``False`` it precomputes the neighbhorhoods in adj,
        if ``False`` it skips the computation and reads it fromt the input
    neighborhoods : DataFrame
        DataFrame of neighbhoord indices. Required if precomputed is ``True``

    Returns
    -------
    DataFrame
        with index, centers to be considered and columns the sum, mean, standard deviation and
        standard error of the mean of the values in that neighborhood.

    See Also
    --------
    [neighborhood_indices] (network_local.md#src.connalysis.network.local.neighborhood_indices):
    Function to precompute the neighborhood_indices that can be used if precomputed is set ``True``.
    Precomputing the neighborhoods would increase efficiency if multiple properties are averaged across neighborhoods.
    """

    # Single value functions for DataFrames
    def append_center(x):
        # To include center in the computation
        return np.append(x["center"], x["neighbors"])

    def mean_nbd(nbd_indices, v):
        df = v[nbd_indices]
        return [np.nansum(df), np.nanmean(df), np.nanstd(df), stats.sem(df, nan_policy="omit")]

    # Get neighborhoods
    if precomputed:
        assert isinstance(neighborhoods,
                          pd.Series), "If precomputed a Series of neighbhoords indexed by their center must be provided"
    else:
        assert (adj is not None), "If not precomputed and adjancecy matrix must be provided"
        neighborhoods = neighborhood_indices(adj, pre=pre, post=post, all_nodes=all_nodes, centers=centers)
    centers = neighborhoods.index
    if include_center:
        neighborhoods = neighborhoods.reset_index().apply(append_center, axis=1)
    else:
        neighborhoods = neighborhoods.reset_index(drop=True)
    stat_vals = pd.DataFrame.from_records(neighborhoods.map(lambda x: mean_nbd(x, values)),
                                          columns=["sum", "mean", "std", "sem"])
    stat_vals["center"] = centers
    return stat_vals.set_index("center")

def edge_stats_participation(participation, vals, condition=operator.eq, dims=None):
    """ Get statistics of the values in vals across edges filtered using edge participation

    Parameters
    ----------
    participation : DataFrame
        DataFrame of edge participation with index the edges of an NxN matrix to consider,
        columns are dimensions and values are edge participation.
    values : Series
        pandas Series with index the edges of the NxN matrix of which edge participation has been computed
        and vals the values on that edge to be averaged.
    condition : operator
        operator with which to filter the nodes. The default ``operator.eq`` filters nodes such that their maximal
        dimension of edge participation is a given value.
        Alternatively, ``operator.ge`` filters edges such that their maximal dimension of node participation is at least a given
        value.
    dims : iterable
        dimensions for which to run the analysis, if ``None`` all the columns of participation will be analyzed

    Returns
    -------
    DataFrame
        with index, the dimensions for which the analysis have been run and columns the statistics of the values in vals
        where the nodes have been grouped according to the condition given.
    """
    par_df = participation.copy()
    if dims is None:
        dims = par_df.columns
    vals = vals.rename("values")
    par_df["max_dim"] = (par_df > 0).sum(axis=1)  # maximal dimension an edge is part of. Note that edge participation in dimension 0 is 0
    stats_vals = {}
    for dim in dims:
        mask = condition(par_df.max_dim, dim)
        c = pd.DataFrame(vals.loc[par_df[mask].index], columns=["values"])
        c["weight"] = par_df[mask][dim]
        w_mean = c.apply(np.product, axis=1).sum() / (c["weight"].sum())
        stats_vals[dim] = (c.shape[0],  # Number of nodes fulfilling the condition
                           np.nanmean(c["values"]),
                           np.nanstd(c["values"]),
                           stats.sem(c["values"], nan_policy="omit"),
                           w_mean  # mean weighted by participation
                           )
    stats_vals = pd.DataFrame.from_dict(stats_vals, orient="index",
                                        columns=["counts", "mean", "std", "sem", "weighted_mean"])
    stats_vals.index.name = "dim"
    return stats_vals.drop(0)



'''###POTENTIALLY USEFUL OR DELETE

def edge_par_filter(edge_par_df, dims = None, id_mapping=None):
    # TODO: Get edge_par inside function and take adj instead?
    # Build df with index edges indexed by id_mapping and columns edge_participation per dimension and max_dimension
    # If id_mapping == None nodes are indexed by range(number of nodes) and edges accordingly
    # dims = None (do all dimensions), assert dims are in columns
    # out df
    pass
def node_par_filter(node_par_df, dims = None, id_mapping=None):
    # TODO: Get node_par inside function and take adj instead?
    # Build df with index nodex indexed by id_mapping and columns edge_participation per dimension and max_dimension
    # If id_mapping == None nodes are indexed by range(number of nodes)
    # dims = None (do all dimensions), assert dims are in columns
    # out df
    pass

def slist_edge_filter(slists, edge_type, dims=None, mapping_id=None)
    # TODO: Get slists inside function and take adj instead?
    # TODO: Multiple edge_types in one go?
    # Build df with index nodex indexed by id_mapping and columns edge_participation per dimension and max_dimension
    # If id_mapping == None nodes are indexed by range(number of nodes)
    # dims = None (do all dimensions) assert dims are in index
    # edge_type: function that takes dim and gives source an target
    # out dict
    pass

def slist_node_filter(slists, position, dims=None, mapping_id=None)
    # TODO: Get slists inside function and take adj instead?
    # TODO: Multiple positions in one go.
    # Build df with index nodex indexed by id_mapping and columns edge_participation per dimension and max_dimension
    # If id_mapping == None nodes are indexed by range(number of nodes)
    # dims = None (do all dimensions) assert dims are in index
    # position: position of node in simplex
    # out dict
    pass


#### STUFF FROM MICRONS PAPER

#Restriction to valid gids i.e., gids that are both in the connectome and that have recorded activity

def get_valid_gid(conn_gids, record_gids):
    # Return set of gids for which both included in the connectome and whose activity is measured
    ids, counts=np.unique(record_gids,return_counts=True)
    record_gids=ids[counts<=1]
    return np.intersect1d(conn_gids, record_gids)

def add_row_col_gid_and_mdim(df_par, all_gids, map_gids):
    # Add row and column gids to indexing to df of edge participation
    df=df_par.copy()
    df['max_dim']=((df>=1).sum(axis=1))
    df=df.set_index(keys=pd.MultiIndex.from_tuples(df.index)).reset_index(names=["row", "col"])
    index=pd.MultiIndex.from_tuples(zip(map_gids.loc[df.row].to_numpy(),
                                    map_gids.loc[df.col].to_numpy()))
    return df.set_index(index)

def get_valid_edges_mask(df, valid_gids):
    # Get mask of edge particiapation on edges between valid gids
    #df=df.reset_index(names=["row_gid", "col_gid"])
    df=df.reset_index()
    return np.logical_and(np.isin(df.level_0, valid_gids), np.isin(df.level_1, valid_gids))

# Format correlation matrix
def build_corr_df(corr, record_gids, valid_gids):
    # Store correlation as a Series with index the corresponding edge
    # Restrict correlation matrix to valid gids
    mask=np.isin(record_gids, valid_gids)
    corr=corr[np.ix_(mask, mask)]; N=corr.shape[0]
    # Get off diagonal entries
    id_up=np.triu_indices(N,k=1); id_down=np.tril_indices(N,k=-1)
    off_diag=(np.concatenate([id_up[0], id_down[0]]),
              np.concatenate([id_up[1], id_down[1]]))
    # Index by gids of entries
    index=pd.MultiIndex.from_tuples(zip(valid_gids[off_diag[0]],valid_gids[off_diag[1]]))
    return pd.Series(corr[off_diag], index=index)

# Analysis based on edge participation
def format_edge_list(s_list, source, target, map_gids):
    # Get list of edges in s_list from source to target indexed by their gids
    df=pd.DataFrame(s_list[:, [source,target]], columns=["row", "col"])
    index=pd.MultiIndex.from_tuples(zip(map_gids.loc[df.row].to_numpy(),
                                    map_gids.loc[df.col].to_numpy()))
    return df.set_index(index)

def stats_edge_par(edge_df, corr_df, condition):
    # Get stats for edges with ege participation according to condition
    # condition operator.eq (max dimension equals a given value)
    # condition operator.ge (max dimension is at least a certain value)
    dims=np.arange(1, edge_df.columns[-2]+1)
    stats={}
    for dim in dims:
        mask=condition (edge_df.max_dim, dim)
        c=pd.DataFrame(corr_df.loc[edge_df[mask].index], columns=["correlation"])
        c["weight"]=edge_df[mask][dim]
        w_mean=c.apply(np.product, axis=1).sum()/(c["weight"].sum())
        stats[dim]=(c.shape[0], c["correlation"].mean(), sem(c["correlation"]), w_mean)
    return pd.DataFrame.from_dict(stats, orient="index", columns=["counts", "mean", "sem", "weighted_mean"])

# Analysis based on simplex list
def last_edge(dim):
    return dim-1, dim

def stats_edge_position(simplex_lists, corr_df,map_gids,valid_gids, edge_selection=last_edge):
    # Get stats for edges in a simplex in the position given by edge_selection
    dims=simplex_lists.index[1:]
    stats={}
    for dim in tqdm(dims):
        source, target= edge_selection(dim)
        edges=format_edge_list(simplex_lists[dim], source,target,map_gids)
        edges=edges[get_valid_edges_mask(edges, valid_gids)]
        c=corr_df.loc[edges.index]
        stats[dim]=(c.shape[0], c.mean(), sem(c))
    return pd.DataFrame.from_dict(stats, orient="index", columns=["counts", "mean", "sem"])'''