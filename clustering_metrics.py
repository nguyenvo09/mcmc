import numpy as np
import math
import scipy.sparse as sparse
import networkx as nx

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
    a = (sum1 - n)/2.0
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

def entropy(nums):
    '''
    Entropy of a list of integer numbers.
    :param nums:
    :return: Entropy of @nums
    '''
    z = np.bincount(nums)
    N = len(nums)
    assert nums.shape == (N, )
    ent = 0.0
    assert sum(z) == N
    for e in z:
        if e != 0:
            p = float(e) / N
            ent += p*math.log(p)
    assert ent <= 0
    ent = -ent

    assert ent >=0
    return ent
def computeNMI(clusters, classes):
    '''
    @ref: Page 359, An Introduction to Information Retrieval (by Stanford)
    compute Normalized Mutual Information from
    :param partitions: dictionary of vertex u and its community
    :param groundtruth: true community of vertex u.
    :return: Normalized Mutual Information
    Example:
    clusters = np.array([1, 2, 3, 4, 5, 6,    7,  8,  9,  10,  11,  12,  13, 14, 15, 16, 17])
    classses = np.array([18, 2, 3, 4, 5, 6,    7,  8,  9,  10,  11,  12,  13, 14, 15, 16, 17])
    NMI = computeNMI(clusters, classses)
    print NMI
    assert abs(NMI - 1.0) <= 1e-10
    '''

    assert clusters.shape == classes.shape
    A = np.c_[(clusters, classes)]
    A = np.array(A)
    N = A.shape[0]
    assert A.shape == (N, 2)

    H_clusters = entropy(A[:, 0])
    H_classes = entropy(A[:, 1])
    # print H_clusters
    # print H_classes
    # assert N == 17
    NMI = 0.0
    for k in np.unique(A[:, 0]):
        # get elements in second column that have first column equal to j
        z = A[A[:, 0] == k, 1]
        len_wk = len(z)
        t = A[:, 1]
        #for each unique class in z
        for e in np.unique(z):

            wk_cj=len(z[z==e])
            len_cj=len(t[t == e])
            assert wk_cj <= len_cj
            numerator= (float(wk_cj) / float(N)) * math.log( (N*wk_cj) / float(len_wk * len_cj)  )
            NMI += numerator
    NMI /= float((H_clusters + H_classes) * 0.5)

    assert (NMI > 0.0 or abs(NMI) < 1e-10) and (NMI < 1.0 or abs(NMI - 1.0) < 1e-10)
    return NMI

def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    @link: http://www.caner.io/purity-in-python.html
    :param clusters: the cluster assignments array
    :type clusters: numpy.array

    :param classes: the ground truth classes
    :type classes: numpy.array

    :returns: the purity score
    :rtype: float
    Example:
    clusters = np.array([1, 1, 1, 1, 1, 1, 2,2,2,2,2,2, 3,3,3,3,3])
    classses = np.array([1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 4, 5, 1, 1, 5, 5, 5])
    a = purity_score(clusters, classses)
    print a
    >> 1.0
    """
    assert clusters.shape == classes.shape
    A = np.c_[(clusters, classes)]
    # print A
    N,  = clusters.shape
    assert A.shape == (N, 2)
    n_accurate = 0.

    for j in np.unique(A[:, 0]):
        #get elements in second column that have first column equal to j
        z = A[A[:, 0] == j, 1]
        # find the value that appear most frequently of @classes.
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]

def modularity(partition, graph) :
    """Compute the modularity of a partition of a graph based on equation
    https://www.cs.umd.edu/class/fall2009/cmsc858l/lecs/Lec10-modularity.pdf

    @partition : dict the partition of the nodes, i.e a dictionary where keys are their nodes and values the communities
    @graph : networkx.Graph the networkx graph which is decomposed
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community structure in networks. Physical Review E 69, 26113(2004).
    Examples:

        edges = [(1, 2, 1), (1, 3, 1), (1, 4, 1),
             (2, 3, 1), (3, 4, 1), (4, 5, 1),
             (4, 6, 1), (5, 6, 1), (5, 7, 1),
             (5, 8, 1), (6, 7, 1), (6, 8, 1),
             (7, 8, 1), (7, 9, 1)]
        membership = {1: 1, 2: 1, 3: 1, 4: 1,
                  5: 2, 6: 2, 7: 2, 8: 2,
                  9: 3}
        G = nx.Graph()
        for e in edges:
            G.add_edge(e[0], e[1], weight=e[2])
        print modularity(membership, G)
        >>
    --------
    """

    inc = {}
    deg = {}
    no_links = graph.size(weight='weight')

    assert type(graph) == nx.Graph, 'Bad graph type, use only non directed graph'

    if abs(no_links) <= 1e-10 :
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph :
        com = partition[node]
        #computing cross-edge-weight and inner-edge-weight
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight = 'weight')
        inc[com] = inc.get(com, 0.)
        #compute inner-edge weight sum.
        for neighbor, datas in graph[node].items() :
            weight = datas.get("weight", 1.0)
            if partition[neighbor] == com :
                assert neighbor != node, 'There is no self-looping, please!!!'
                inc[com] = inc.get(com, 0.) + float(weight)

    modularity_ = 0.
    cnt = 0
    for com in set(partition.values()) :
        assert com in deg
        cnt += 1
        assert deg[com] >= inc[com]
        a_i = deg[com] / float(2.0* no_links)
        modularity_ += (inc[com] / float(2*no_links)) - a_i*a_i


    return (modularity_, cnt)


def Ncut(partition, graph):
    '''
    k=number of partitions. k can be called the number of communities.
    Normalized cut of a graph given community-membership of vertices in a graph G(V,E) is:
    Ncut(G)=1.0/k * \sum_{i=1}^{k} cut(C_i, complemented_C_i) / vol(C_i)

    C_i is a community in G(V,E).
    In unweighted graph: Vol(C_i) = sum degree of all vertices in C_i = 2* edges that have at least one point in C_i.
                         Cut(C_i, rest) = the number of edges (u,v) such that u \in @C_i and v \in @rest
                         @rest: mean that vertices that in V but not in C_i


    In weighted graph G(V, E): Vol(C_i) = 2 * (sum of weight of all edges (u,v) that have u\in C_i OR v \in C_i)
                               Cut(C_i, rest) = the sum of weight of edges (u,v): u\in @C_i and v \in @rest

    In http://www.cs.cmu.edu/~jshi/papers/pami_ncut.pdf Equation 1 and 2:
    Note: assoc(A,V) is exactly same to Vol(A) where A is a community (i.e. A is C_i)
    V is set of vertices in G(V,E).
    :param partition: a dictionary containing communityship of |V| vertices. key is a vertex, value is community
    :param graph: networkX graph.
    :return: normalized cut of graph in range [0-1] and the number of communities in @partition.
     Example:
     edges = [(1, 2, 0.1), (1, 3, 0.1), (1, 4, 0.1),
             (2, 3, 0.1),
             (3, 4, 0.1),
             (4, 5, 0.1), (4, 6, 0.1),
             (5, 6, 0.1), (5, 7, 0.1), (5, 8, 0.1),
             (6, 7, 0.1), (6, 8, 0.1),
             (7, 8, 0.1), (7, 9, 0.3)]
            membership = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,  9: 2}
            G = nx.Graph()
            for e in edges:
                G.add_edge(e[0], e[1], weight=e[2])
            print Ncut(membership, G)
            >> (0.5517241379, 2)
    '''

    assert type(graph) == nx.Graph, 'Input graph must be undirected networkX graph'
    assert len(partition) == len(graph.nodes()), 'all nodes must have a partition'

    cross_edges_weight = {}
    cross_edges_inner_edges_weight = {}
    #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.Graph.size.html
    '''Return |E| if unweighted, sum(w(u,v)) if weighted'''
    no_links = graph.size(weight='weight')

    if abs(no_links) <= 1e-10 :
        raise ValueError("A graph without link has an undefined normalized cut")

    for node in graph :
        com = partition[node]
        #summing degree of @node or sum weight w(v, node) of v \in Neb(node)
        #computing Vol(com)
        cross_edges_inner_edges_weight[com] = cross_edges_inner_edges_weight.get(com, 0.) + graph.degree(node, weight = 'weight')

        #Computing Cut(com, rest)
        #we loop though all edges (u,v) such that u=node, and v \in @rest_nodes
        for neighbor, datas in graph[node].items() :
            weight = datas.get("weight", 1.0)
            cross_edges_weight[com] = cross_edges_weight.get(com, 0.)
            assert neighbor in partition, 'missing community!!!'
            if partition[neighbor] != com:
                cross_edges_weight[com] = cross_edges_weight.get(com, 0.) + weight
    ncut_ = 0.
    cnt = 0
    assert len(cross_edges_inner_edges_weight) == len(cross_edges_weight)
    #loop through set of communities
    for com in set(partition.values()) :
        assert com in cross_edges_inner_edges_weight
        assert cross_edges_inner_edges_weight[com] >= cross_edges_weight[com]
        cnt += 1
        ncut_ += cross_edges_weight[com] / float(cross_edges_inner_edges_weight[com])
    ncut_ /= float(cnt)
    return (ncut_, cnt)

if __name__ == '__main__':
    print 'ben'