import numpy as np
import math

def entropy(nums):
    z = np.bincount(nums)
    N = len(nums)
    assert nums.shape == (N, )
    ent = 0.0
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
    :return:
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

def test():
    clusters = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    classses = np.array([1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 4, 5, 1, 1, 5, 5, 5])
    print computeNMI(clusters, classses)

    clusters = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    classses = np.array([4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7])
    NMI= computeNMI(clusters, classses)
    print NMI
    assert abs(NMI - 1.0) <= 1e-10

    clusters = np.array([1, 2, 3, 4, 5, 6,    7,  8,  9,  10,  11,  12,  13, 14, 15, 16, 17])
    classses = np.array([18, 2, 3, 4, 5, 6,    7,  8,  9,  10,  11,  12,  13, 14, 15, 16, 17])
    NMI = computeNMI(clusters, classses)
    print NMI
    assert abs(NMI - 1.0) <= 1e-10

    clusters = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    classses = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])
    NMI = computeNMI(clusters, classses)
    print NMI
    assert abs(NMI - 0.0) <= 1e-10

    clusters = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3])
    classses = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    NMI = computeNMI(clusters, classses)
    print NMI
    assert abs(NMI - 0.0) <= 1e-10

if __name__ == '__main__':
    print 'here'
    test()