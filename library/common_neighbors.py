#Moved to src/connalysis/network
import numpy as np
import pandas as pd


def __make_expected_distribution_model_first_order__(adj, direction="efferent"):
    from scipy.stats import hypergeom
    if direction == "efferent":
        N = adj.sum(axis=1).mean()
        M = adj.shape[1]
    elif direction == "afferent":
        N = adj.sum(axis=0).mean()
        M = adj.shape[0]
    else:
        raise ValueError("Unknown direction: {0}".format(direction))
    expected = hypergeom(M, N, N)
    return expected


def distribution_number_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    adj = adj.tocsc().astype(int)
    if direction == "efferent":
        cn = adj * adj.transpose()
    elif direction == "afferent":
        cn = adj.transpose() * adj
    else:
        raise ValueError("Unknown direction: {0}".format(direction))
    cn = np.array(cn.todense())
    cn = cn[np.triu_indices_from(cn, 1)]
    bins = np.arange(0, cn.max() + 2)
    return pd.Series(np.histogram(cn, bins=bins)[0], index=bins[:-1])


def normalized_distribution_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    expected = __make_expected_distribution_model_first_order__(adj, direction=direction)
    expected = expected.pmf(data.index) * data.sum()
    expected = pd.Series(expected, index=data.index)
    return (data - expected) / (data + expected)


def overexpression_of_common_neighbors(adj, neuron_properties=None, direction="efferent"):
    data = distribution_number_of_common_neighbors(adj, neuron_properties, direction=direction)
    data_mean = (data.index.values * data.values).sum() / data.values.sum()
    ctrl = __make_expected_distribution_model_first_order__(adj, direction=direction)
    ctrl_mean = ctrl.mean()
    return (data_mean - ctrl_mean) / (data_mean + ctrl_mean)


def common_neighbor_weight_bias(adj, neuron_properties=None, direction="efferent"):
    adj_bin = (adj.tocsc() > 0).astype(int)
    if direction == "efferent":
        cn = adj_bin * adj_bin.transpose()
    elif direction == "afferent":
        cn = adj_bin.transpose() * adj_bin

    return np.corrcoef(cn[adj > 0],
                          adj[adj > 0])[0, 1]


def common_neighbor_connectivity_bias(adj, neuron_properties=None, direction="efferent",
                                      cols_location=None, fit_log=False):
    import statsmodels.formula.api as smf
    from patsy import ModelDesc
    from scipy.spatial import distance

    if adj.dtype == bool:
        adj = adj.astype(int)

    if direction == "efferent":
        cn = adj * adj.transpose()
    elif direction == "afferent":
        cn = adj.transpose() * adj

    input_dict = {"CN": cn.toarray().flatten(),
                  "Connected": adj.astype(bool).toarray().flatten()}
    
    if fit_log:
        input_dict["CN"] = np.log10(input_dict["CN"] + fit_log)
    formula_str = "CN ~ Connected"
    if cols_location is not None:
        formula_str = formula_str + " + Distance"
        dmat = distance.squareform(distance.pdist(neuron_properties[cols_location].values))
        input_dict["Distance"] = dmat.flatten()
    sm_model = ModelDesc.from_formula(formula_str)

    sm_result = smf.ols(sm_model, input_dict).fit()

    pval = sm_result.pvalues.get("Connected[T.True]", 1.0)
    mdl_intercept = sm_result.params["Intercept"]
    mdl_added = sm_result.params.get("Connected[T.True]", 0.0)
    mdl_distance = sm_result.params.get("Distance", 0.0)
    return pval, mdl_added / mdl_intercept, 100 * mdl_distance / mdl_intercept
