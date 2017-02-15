import numpy as np


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

def test():
    clus = np.array([1, 4, 4, 4, 4, 4, 3, 3, 2, 2, 3, 1, 1])
    clas = np.array([5, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 5, 2])
    print purity_score(clus, clas)

    clusters = np.array([1, 1, 1, 1, 1, 1, 2,2,2,2,2,2, 3,3,3,3,3])
    classses = np.array([1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 4, 5, 1, 1, 5, 5, 5])
    a = purity_score(clusters, classses)
    print 'here: ', a
    x= 1.0/17 * (5 + 4 + 3)
    assert abs(a - x) < 1e-10

if __name__ == '__main__':
    test()