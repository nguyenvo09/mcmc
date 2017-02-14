import numpy as np
import networkx as nx

'''
Computing modularity of a graph G(V, E), each node 'u \in V' will belong to a community.
Details:
Implementation of http://perso.crans.org/aynaud/communities/index.html is wrong in modularity function
where there is self-looping edge. This does not matter if our graph has no self-loop edge. This implemention,
however, when applied to Louvain which includes re-creating graph after first-pass of finding community, leads to
mistake in computing modularity!!!!

@author: ben
'''

def modularity(partition, graph) :
    """Compute the modularity of a partition of a graph based on equation
    https://www.cs.umd.edu/class/fall2009/cmsc858l/lecs/Lec10-modularity.pdf

    @partition : dict the partition of the nodes, i.e a dictionary where keys are their nodes and values the communities
    @graph : networkx.Graph the networkx graph which is decomposed
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community structure in networks. Physical Review E 69, 26113(2004).
    Examples
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

                # if neighbor == node :
                #     inc[com] = inc.get(com, 0.) + float(weight)
                # else :
                #     # This is because inner-edges will be considered twice. While cross-edges are considered only once!!!!
                #     inc[com] = inc.get(com, 0.) + float(weight)

    modularity_ = 0.
    cnt = 0
    # print 'No_edges: ', no_links
    for com in set(partition.values()) :
        assert com in deg
        cnt += 1
        # modularity_ += (inc[com] / float(no_links)) - (deg[com] / (2.0*no_links) )**2

        # This is because inner-edges will be considered twice. While cross-edges are considered only once!!!!s
        assert deg[com] >= inc[com]
        # deg[com] = deg[com] - inc[com]
        # print 'Community: ', com, 'Inner-edges: ', inc[com], ' Inner+Outer-edges: ', deg[com]
        a_i = deg[com] / float(2.0* no_links)
        modularity_ += (inc[com] / float(2*no_links)) - a_i*a_i


    return (modularity_, cnt)

def modularity_way2(partition, graph):
    '''
    Second implementation of modularity of G(V,E) based on dictionary of partition
    :param partition: dictionary of partition of graph. Each node belongs to a community.
    :param graph: undirected graph G(V,E)
    :return:
    '''
    no_links = graph.size(weight='weight')

    assert type(graph) == nx.Graph, 'Bad graph type, use only non directed graph'
    if abs(no_links) <= 1e-10 :
        raise ValueError("A graph without link has an undefined modularity")

    rev_partitions = {}
    for vertex, community in partition.iteritems():
        rev_partitions[community] = rev_partitions.get(community, {})
        assert vertex not in rev_partitions[community]
        rev_partitions[community][vertex] = 1
    DEBUG = True
    if DEBUG:
        r = 0
        for community,vertices in rev_partitions.iteritems():
            r += len(vertices)
        assert r == len(graph.nodes()), 'Something wrong!!!!'
    Q = 0.0
    cnt = 0
    for community, dict_vertices in rev_partitions.iteritems():
        vertices = dict_vertices.keys()
        cnt += 1
        count_loop = 0
        total_edge = 0
        # print vertices
        dict_edges = {}
        '''We need to loop all possible pair of vertices in community k even if there is no edge between u and v. A_{u,v}=0!!!
        Oh my gosh!!!!'''
        for i in xrange(len(vertices)):
            for j in xrange( len(vertices)):
                count_loop += 1
                u = vertices[i]
                v = vertices[j]
                d_i = graph.degree(u, weight='weight')
                d_j = graph.degree(v, weight='weight')
                t = 0
                if u in graph[v] or v in graph[u]:
                    t = graph[u][v]['weight']
                    assert graph[u][v]['weight'] == graph[v][u]['weight']
                    total_edge += 1
                x = (t - d_i * d_j / (2.0 * no_links))
                Q += x

    Q = Q/(2.0 * no_links)
    return (Q, cnt)

def modularity_way3(partition, graph):
    '''Q = 1/(2m)* Trace(S^T.dot(B).dot(S)
    Where S.shape = (N,K) and B.shape = (N,N)
    '''
    N = len(partition) #the number of vertices.
    K = len(set(partition)) #the number of community
    m = graph.size(weight='weight')
    #creating matrix of communities
    S = np.zeros((N, K))
    for vertex, comm in partition.iteritems():
        v = vertex - 1
        cc = comm - 1
        assert v >= 0 and v < N
        assert cc >= 0 and cc < K
        S[v, cc] = 1.0
    assert abs(np.sum(S) - N) <= 1e-10, 'There must be N entries with 1.0'
    #creating vector of degree of each vertex
    D = [0 for i in xrange(N)]
    for node in graph:
        u = node-1
        assert u>=0 and u<N
        D[u] = graph.degree(node, weight='weight')
    D = np.array(D)
    D = D[np.newaxis, :].T
    assert D.shape == (N, 1)
    # print 'Degree vector: ', D
    #creating matrix A
    A = np.zeros((N, N))
    for edge in graph.edges(data=True):
        u,v,info = edge
        u-=1
        v-=1
        w = info['weight']
        assert u>=0 and u<N and v >=0 and v<N
        A[u][v] = float(w)
        A[v][u] = float(w)
    # print A
    assert abs(np.sum(A) - 2*m) < 1e-10
    ###############################################
    B = A - 1.0/(2.0*m) * D.dot(D.T)
    assert B.shape == (N, N)
    #based on book CommunityDetectionAndMining HuanLiu
    Q = 1.0/(2.0*m) * np.trace(S.T.dot(B).dot(S))
    return (Q, K)

def computeModularity(graph_file, no_comms):
    fin=open(graph_file, 'rb')
    partitions = {}
    G = nx.Graph()
    for line in fin:
        line=line.replace('\n', '')
        u,v,comU,comV = line.split()
        partitions[u] = comU
        partitions[v] = comV
        G.add_edge(u,v)
    modu,no_comm= modularity(partition=partitions, graph=G)
    assert no_comm == no_comms
    return modu, no_comm

def test_modularity1():
    edges = [(1, 2, 0.1), (1, 3, 0.1), (1, 4, 0.1),
             (2, 3, 0.1),
             (3, 4, 0.1),
             (4, 5, 0.3), (4, 6, 0.4),
             (5, 6, 0.1), (5, 7, 0.1), (5, 8, 0.1),
             (6, 7, 0.1), (6, 8, 0.1),
             (7, 8, 0.1), (7, 9, 0.5)]

    edges = [(1, 2, 1), (1, 3, 1), (1, 4, 1),
             (2, 3, 1), (3, 4, 1), (4, 5, 1),
             (4, 6, 1), (5, 6, 1), (5, 7, 1),
             (5, 8, 1), (6, 7, 1), (6, 8, 1),
             (7, 8, 1), (7, 9, 1)]
    degs = (1 + 3 + 4 + 4 + 4 + 4 + 3 + 2 + 3)
    membership = {1: 1, 2: 1, 3: 1, 4: 1,
                  5: 2, 6: 2, 7: 2, 8: 2,
                  9: 3}
    assert len(membership) == 9
    assert len(set(membership.values())) == 3
    assert degs % 2 == 0
    assert len(edges) == degs / 2, '%s vs %s' % (len(edges), degs / 2)
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])

    modularity_0, no_comm_1 = modularity_way3(partition=membership, graph=G)
    modularity_1, no_comms = modularity(membership, G)
    modularity_2, no_comms2 = modularity_way2(membership, G)

    print modularity_0, modularity_1, modularity_2

    assert no_comms == no_comms2 and no_comms2 == 3
    assert abs(modularity_1 - modularity_2) < 1e-10, 'Mismatched value two implementations: %s vs. %s' % (modularity_1, modularity_2)

def test_modularity_of_networkx():

    edges = [(1, 2, 1), (1, 3, 1), (1, 4, 1),
             (2, 3, 1), (3, 4, 1), (4, 5, 1),
             (4, 6, 1), (5, 6, 1), (5, 7, 1),
             (5, 8, 1), (6, 7, 1), (6, 8, 1),
             (7, 8, 1), (7, 9, 1), (1,1,1)]
    G = nx.Graph()
    membership = {1: 1, 2: 1, 3: 1, 4: 1,
                  5: 2, 6: 2, 7: 2, 8: 2,
                  9: 3}
    import community
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])
    print community.modularity(membership, G)

if __name__ == '__main__':
    print 'here'
    test_modularity_of_networkx()
    test_modularity1()
