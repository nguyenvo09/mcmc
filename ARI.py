import numpy as np
import scipy.sparse as sparse

def ARI(cluster, classes):
    '''Implementation of Adjusted Rand Index'''

def contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0], norm=True):
    """Return the contingency table for all regions in matched segmentations.

    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg : list of int, optional
        Values to ignore in `seg`. Voxels in `seg` having a value in this list
        will not contribute to the contingency table. (default: [0])
    ignore_gt : list of int, optional
        Values to ignore in `gt`. Voxels in `gt` having a value in this list
        will not contribute to the contingency table. (default: [0])
    norm : bool, optional
        Whether to normalize the table so that it sums to 1.

    Returns
    -------
    cont : scipy.sparse.csc_matrix
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `seg` and `j` in `gt`. (Or the proportion of such voxels
        if `norm=True`.)
    """
    segr = seg.ravel()
    gtr = gt.ravel()
    ij = np.vstack((segr, gtr))
    selector = np.ones(segr.shape, np.bool)
    data = np.ones(len(gtr))
    for i in ignore_seg:
        selector[segr == i] = 0
    for j in ignore_gt:
        selector[gtr == j] = 0
    ij = ij[:, selector]
    data = data[selector]
    cont = sparse.coo_matrix((data, ij)).tocsc()
    if norm:
        cont /= float(cont.sum())
    return cont

def rand_values(cont_table):
    """Calculate values for Rand Index and related values, e.g. Adjusted Rand.

    Parameters
    ----------
    cont_table : scipy.sparse.csc_matrix
        A contingency table of the two segmentations.

    Returns
    -------
    a, b, c, d : float
        The values necessary for computing Rand Index and related values. [1, 2]

    References
    ----------
    [1] Rand, W. M. (1971). Objective criteria for the evaluation of
    clustering methods. J Am Stat Assoc.
    [2] http://en.wikipedia.org/wiki/Rand_index#Definition on 2013-05-16.
    """
    n = cont_table.sum()
    sum1 = (cont_table.multiply(cont_table)).sum()
    sum2 = (np.asarray(cont_table.sum(axis=1)) ** 2).sum()
    sum3 = (np.asarray(cont_table.sum(axis=0)) ** 2).sum()
    a = (sum1 - n)/2.0;
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2
    return a, b, c, d

def adj_rand_index(x, y=None):
    """Return the adjusted Rand index.

    The Adjusted Rand Index (ARI) is the deviation of the Rand Index from the
    expected value if the marginal distributions of the contingency table were
    independent. Its value ranges from 1 (perfectly correlated marginals) to
    -1 (perfectly anti-correlated).

    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that is *not* normalised to sum to 1.

    Returns
    -------
    ari : float
        The adjusted Rand index of `x` and `y`.
    """
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    nk = a+b+c+d
    return (nk*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(
        nk**2 - ((a+b)*(a+c) + (c+d)*(b+d)))

def rand_index(x, y=None):
    """Return the unadjusted Rand index. [1] Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that is *not* normalised to sum to 1.
    Returns
    -------
    ri : float
        The Rand index of `x` and `y`.
    References
    ----------
    [1] WM Rand. (1971) Objective criteria for the evaluation of clustering methods.
    """
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    return (a+d)/(a+b+c+d)

def test():
    clusters = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    classses = np.array([1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 4, 5, 1, 1, 5, 5, 5])
    print adj_rand_index(clusters, classses)
    print rand_index(clusters, classses)

    clusters = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    classses = np.array([1, 2, 1, 2, 2, 2, 4, 4, 4, 4])
    print adj_rand_index(clusters, classses)
    print rand_index(clusters, classses)

if __name__ == '__main__':
    print 'here'
    test()