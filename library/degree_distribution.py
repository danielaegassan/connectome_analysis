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
    gc = gini_curve(m, direction=direction)
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

    edge_counter = lambda i: (degrees >= i).sum() * ((degrees >= i).sum() - 1)
    mat_counter = lambda i: m[numpy.ix_(degrees >= i, degrees >= i)].sum()
    ret = (numpy.array([mat_counter(i) for i in udegrees]).astype(float)
            / numpy.array([edge_counter(i) for i in udegrees]))
    return pandas.Series(ret, index=ret_x)


def _analytical_expected_rich_club_curve(m, direction='efferent'):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    indegree = m.sum(axis=0)
    outdegree = m.sum(axis=1)

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
    return pandas.DataFrame.from_dict({"mean": numpy.array(res_mn),
                                       "std": numpy.array(res_sd)},
                                       index=udegrees)


def normalized_rich_club_curve(m, nrn, direction='efferent', normalize='std', **kwargs):
    assert m.dtype == bool, "This function only works for binary matrices at the moment"
    data = rich_club_curve(m, direction=direction)
    A = data.index.values
    B = data.values
    ctrl = _analytical_expected_rich_club_curve(m, direction=direction)
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
