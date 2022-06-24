import numpy
import pandas
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy import sparse


def gini_curve(m, nrn, direction='efferent'):
    m = m.tocsc()
    if direction == 'afferent':
        degrees = numpy.array(m.sum(axis=0)).flatten()
    elif direction == 'efferent':
        degrees = numpy.array(m.sum(axis=1).flatten())
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)
    cs = numpy.cumsum(numpy.flipud(sorted(degrees))).astype(float) / numpy.sum(degrees)
    return pandas.Series(cs, index=numpy.linspace(0, 1, len(cs)))


def gini_coefficient(m, nrn, direction='efferent'):
    gc = gini_curve(m, nrn, direction=direction)
    A = gc.index.values
    B = gc.values
    return numpy.sum(numpy.diff(A) * (B[:-1] + B[1:]) / 2.0)


def _analytical_expected_gini_curve(m, direction='efferent'):
    if direction == 'afferent':
        N = m.shape[0] - 1
        C = m.shape[1] * N
    elif direction == 'efferent':
        N = m.shape[1] - 1
        C = m.shape[0] * N
    P = m.nnz / C
    # Only using degrees, not distribution of weigthts. TODO: Fix that
    x = numpy.arange(N, -1, -1)
    p = binom.pmf(x, N, P)
    A = numpy.cumsum(p) / p.sum()
    B = numpy.cumsum(p * x) / numpy.sum(p * x)
    return pandas.Series(B, index=A)


def normalized_gini_coefficient(m, nrn, direction='efferent'):
    gc = gini_coefficient(m, nrn, direction=direction)
    ctrl = _analytical_expected_gini_curve(m, direction=direction)
    A = ctrl.index.values
    B = ctrl.values
    return 2 * (gc - numpy.sum(numpy.diff(A) * (B[:-1] + B[1:]) / 2.0))


def _bin_degrees(degrees):
    nbins = numpy.maximum(int(len(degrees) * 0.1), numpy.minimum(len(degrees), 30))
    mx = numpy.nanmax(degrees); mn = numpy.nanmin(degrees)
    bins = numpy.linspace(mn, mx + 1E-6 * (mx - mn), nbins + 1)
    degrees = numpy.digitize(degrees, bins=bins) - 1
    udegrees = numpy.arange(nbins)
    ret_x = 0.5 * (bins[:-1] + bins[1:])
    return ret_x, udegrees, degrees


def rich_club_curve(m, nrn, direction='efferent'):
    m = m.tocsc()
    if direction == 'afferent':
        degrees = numpy.array(m.sum(axis=0)).flatten()
    elif direction == 'efferent':
        degrees = numpy.array(m.sum(axis=1)).flatten()
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)

    if m.dtype == bool:
        udegrees = numpy.arange(1, degrees.max() + 1)
        ret_x = udegrees
    else:
        ret_x, udegrees, degrees = _bin_degrees(degrees)

    edge_counter = lambda i: (degrees >= i).sum() * ((degrees >= i).sum() - 1)  # number of pot. edges
    mat_counter = lambda i: m[numpy.ix_(degrees >= i, degrees >= i)].sum()  # number of actual edges
    ret = (numpy.array([mat_counter(i) for i in udegrees]).astype(float)
            / numpy.array([edge_counter(i) for i in udegrees]))
    return pandas.Series(ret, index=ret_x)


def efficient_rich_club_curve(M, direction="efferent", pre_calculated_degree=None):
    M = M.tocoo()
    shape = M.shape
    M = pandas.DataFrame.from_dict({"row": M.row, "col": M.col})
    if pre_calculated_degree is not None:
        deg = pre_calculated_degree
    elif direction == "efferent":
        deg = M["row"].value_counts()
    elif direction == "afferent":
        deg = M["col"].value_counts()
    else:
        raise ValueError()

    degree_bins = numpy.arange(deg.max() + 2)
    degree_bins_rv = degree_bins[-2::-1]
    nrn_degree_distribution = numpy.histogram(deg.values, bins=degree_bins)[0]
    nrn_cum_degrees = numpy.cumsum(nrn_degree_distribution[-1::-1])
    nrn_cum_pairs = nrn_cum_degrees * (nrn_cum_degrees - 1)

    deg_arr = numpy.zeros(shape[0], dtype=int)
    deg_arr[deg.index.values] = deg.values

    deg = None

    con_degree = numpy.minimum(deg_arr[M["row"].values], deg_arr[M["col"].values])
    M = None
    con_degree = numpy.histogram(con_degree, bins=degree_bins)[0]

    cum_degrees = numpy.cumsum(con_degree[-1::-1])

    return pandas.DataFrame(cum_degrees / nrn_cum_pairs, degree_bins_rv)


def _analytical_expected_rich_club_curve(m, direction='efferent'):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    indegree = numpy.array(m.sum(axis=0))[0]
    outdegree = numpy.array(m.sum(axis=1))[:, 0]

    if direction == 'afferent':
        degrees = indegree
    elif direction == 'efferent':
        degrees = outdegree
    else:
        raise Exception("Unknown value for argument direction: %s" % direction)

    udegrees = numpy.arange(1, degrees.max() + 1)
    edge_counter = lambda i: (degrees >= i).sum() * ((degrees >= i).sum() - 1)
    res_mn = []
    res_sd = []
    for deg in udegrees:
        valid = numpy.nonzero(degrees >= deg)[0]
        i_v = indegree[valid]
        i_sum_all = indegree.sum() - i_v
        i_sum_s = i_v.sum() - i_v
        o_v = outdegree[valid]
        S = numpy.array([hypergeom.stats(_ia, _is, o)
                         for _ia, _is, o in zip(i_sum_all, i_sum_s, o_v)])
        res_mn.append(numpy.sum(S[:, 0]) / edge_counter(deg))
        res_sd.append(numpy.sqrt(S[:, 1].sum()) / edge_counter(deg)) #Sum the variances, but divide the std
    df = pandas.DataFrame.from_dict({"mean": numpy.array(res_mn),
                                       "std": numpy.array(res_sd)})
    df.index = udegrees
    return df


def generate_degree_based_control(M, direction="efferent"):
    """
    A shuffled version of a connectivity matrix that aims to preserve degree distributions.
    If direction = "efferent", then the out-degree is exactly preserved, while the in-degree is 
    approximately preseved. Otherwise it's the other way around.
    """
    if direction == "efferent":
        M = M.tocsr()
        idxx = numpy.arange(M.shape[1])
        p_out = numpy.array(M.mean(axis=0))[0]
    elif direction == "afferent":
        M = M.tocsc()
        idxx = numpy.arange(M.shape[0])
        p_out = numpy.array(M.mean(axis=1))[:, 0]
    else:
        raise ValueError()
    
    for col in range(M.shape[1]):
        p = p_out.copy()
        p[col] = 0.0
        p = p / p.sum()
        a = M.indptr[col]
        b = M.indptr[col + 1]
        M.indices[a:b] = numpy.random.choice(idxx, b - a, p=p, replace=False)
    return M


def _randomized_control_rich_club_curve(m, direction='efferent', n=10):
    res = []
    for _ in range(n):
        m_shuf = generate_degree_based_control(m, direction=direction)
        res.append(efficient_rich_club_curve(m_shuf))
    res = pandas.concat(res, axis=1)
    
    df = pandas.DataFrame.from_dict(
        {
            "mean": numpy.nanmean(rr, axis=1),
            "std": numpy.nanstd(rr, axis=1)
        }    
    )
    df.index = res.index
    return df


def normalized_rich_club_curve(m, nrn, direction='efferent', normalize='std',
                               normalize_with="shuffled", **kwargs):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    data = rich_club_curve(m, nrn, direction=direction)
    A = data.index.values
    B = data.values
    if normalize_with == "analytical":
        ctrl = _analytical_expected_rich_club_curve(m, direction=direction)
    elif normalize_with == "shuffled":
        ctrl = _randomized_control_rich_club_curve(m, direction=direction)
    Ar = ctrl.index.values
    mn_r = ctrl["mean"].values
    sd_r = ctrl["std"].values

    if normalize == 'mean':
        return pandas.Series(B[:len(mn_r)] / mn_r, index=A[:len(mn_r)])
    elif normalize == 'std':
        return pandas.Series((B[:len(mn_r)] - mn_r) / sd_r, index=A[:len(mn_r)])
    else:
        raise Exception("Unknown normalization: %s" % normalize)


def rich_club_coefficient(m, nrn, **kwargs):
    Bn = normalized_rich_club_curve(m, normalize='std', **kwargs).values
    return numpy.nanmean(Bn)
