"""
Functions to analyze activity across simplices.
author: Daniela Egas Santander, last update: 11.2023
"""

import numpy as np
import pandas as pd
from scipy.stats import sem
import conntility
from tqdm import tqdm


# Restriction to valid gids i.e., gids that are both in the connectome and that have recorded activity

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
    return pd.DataFrame.from_dict(stats, orient="index", columns=["counts", "mean", "sem"])