import networkx as nx

'''@author: ben'''

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
    :return:
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


def computeNCut(graph_file, no_comms):
    fin = open(graph_file, 'rb')
    partitions = {}
    G = nx.Graph()
    for line in fin:
        line = line.replace('\n', '')
        u, v, comU, comV = line.split()
        partitions[u] = comU
        partitions[v] = comV
        G.add_edge(u, v)
    ncut, no_comm = Ncut(partition=partitions, graph=G)
    assert no_comm == no_comms
    return ncut, no_comm

def testing_ncut1():
    '''Source: slide 24
    http://www.cc.gatech.edu/classes/AY2011/cs7270_spring/7270-community-detection.pptx
        Community Detection and Graph-based Clustering Adapted from Chapter 3 Of Lei Tang and Huan Liu's Book'''
    edges=[(1,2), (1,3), (1,4),
           (2,3),
           (3,4),
           (4,5), (4,6),
           (5,6), (5,7), (5,8),
           (6,7), (6,8),
           (7,8), (7,9)]
    degs = (1+3+4+4+4+4+3+2+3)
    membership = {1:1, 2:1, 3:1, 4:1,
                  5:2, 6:2, 7:2, 8:2, 9:2}
    assert len(membership) == 9
    assert len(set(membership.values())) == 2
    assert degs % 2 == 0
    assert len(edges) == degs/2, '%s vs %s' % (len(edges), degs/2)
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1])
    ncut, no_comms = Ncut(membership, G)
    assert no_comms == 2

    assert abs(ncut - 7.0/48.0) < 1e-10, 'ncut value: %s' % ncut

def testing_ncut2():
    '''Source: slide 24
    http://www.cc.gatech.edu/classes/AY2011/cs7270_spring/7270-community-detection.pptx
    Community Detection and Graph-based Clustering Adapted from Chapter 3 Of Lei Tang and Huan Liu's Book '''
    edges=[(1,2), (1,3), (1,4),
           (2,3),
           (3,4),
           (4,5), (4,6),
           (5,6), (5,7), (5,8),
           (6,7), (6,8),
           (7,8), (7,9)]
    degs = (1+3+4+4+4+4+3+2+3)
    membership = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1,
                  9:2}
    assert len(membership) == 9
    assert len(set(membership.values())) == 2
    assert degs % 2 == 0
    assert len(edges) == degs/2, '%s vs %s' % (len(edges), degs/2)
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1])
    ncut, no_comms = Ncut(membership, G)
    assert no_comms == 2

    assert abs(ncut - 14.0/27.0) < 1e-10, 'ncut value: %s' % ncut

def testing_ncut3():
    '''Source: slide 24
        http://www.cc.gatech.edu/classes/AY2011/cs7270_spring/7270-community-detection.pptx
        Community Detection and Graph-based Clustering Adapted from Chapter 3 Of Lei Tang and Huan Liu's Book '''
    edges = [(1, 2, 0.1), (1, 3, 0.1), (1, 4, 0.1),
             (2, 3, 0.1),
             (3, 4, 0.1),
             (4, 5, 0.1), (4, 6, 0.1),
             (5, 6, 0.1), (5, 7, 0.1), (5, 8, 0.1),
             (6, 7, 0.1), (6, 8, 0.1),
             (7, 8, 0.1), (7, 9, 0.3)]
    degs = (1 + 3 + 4 + 4 + 4 + 4 + 3 + 2 + 3)
    membership = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
                  9: 2}
    assert len(membership) == 9
    assert len(set(membership.values())) == 2
    assert degs % 2 == 0
    assert len(edges) == degs / 2, '%s vs %s' % (len(edges), degs / 2)
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1], weight=e[2])
    ncut, no_comms = Ncut(membership, G)
    assert no_comms == 2

    assert abs(ncut - 0.5517241379) < 1e-10, 'ncut value: %s' % ncut


def testing_ncut4():
    '''Source: slide 24
        http://www.cc.gatech.edu/classes/AY2011/cs7270_spring/7270-community-detection.pptx
        Community Detection and Graph-based Clustering Adapted from Chapter 3 Of Lei Tang and Huan Liu's Book '''
    edges = [(1, 2, 0.1), (1, 3, 0.1), (1, 4, 0.1),
             (2, 3, 0.1),
             (3, 4, 0.1),
             (4, 5, 0.1), (4, 6, 0.1),
             (5, 6, 0.1), (5, 7, 0.1), (5, 8, 0.1),
             (6, 7, 0.1), (6, 8, 0.1),
             (7, 8, 0.1), (7, 9, 0.1)]
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
    ncut, no_comms = Ncut(membership, G)
    assert no_comms == 3

    assert abs(ncut - 41.0/90.0) < 1e-10, 'ncut value: %s' % ncut


def testing_ncut5():
    ''' For weighted graph: Source: slide 24
        http://www.cc.gatech.edu/classes/AY2011/cs7270_spring/7270-community-detection.pptx
        Community Detection and Graph-based Clustering Adapted from Chapter 3 Of Lei Tang and Huan Liu's Book '''
    edges = [(1, 2, 0.1), (1, 3, 0.1), (1, 4, 0.1),
             (2, 3, 0.1),
             (3, 4, 0.1),
             (4, 5, 0.3), (4, 6, 0.4),
             (5, 6, 0.1), (5, 7, 0.1), (5, 8, 0.1),
             (6, 7, 0.1), (6, 8, 0.1),
             (7, 8, 0.1), (7, 9, 0.5)]
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
    ncut, no_comms = Ncut(membership, G)
    assert no_comms == 3

    assert abs(ncut - 65.0/102.0) < 1e-10, 'ncut value: %s' % ncut
if __name__ == '__main__':
    testing_ncut1()
    testing_ncut2()
    testing_ncut3()
    testing_ncut4()
    testing_ncut5()