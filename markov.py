import numpy as np
class MarkovChain(object):
    def __init__(self, u0, P):
        k = P.shape[0]
        assert P.shape == (k,k)
        assert u0.shape == (k,)
        assert abs(np.sum(u0)-1.0) < 1e-10
        assert np.sum(np.abs(np.sum(P, axis=1) - 1.0) < 1e-10) == k #
        self.u0 = u0
        self.P = P
        self.nstates = k
        self.initRanges()
        self.initRangeP()
    def initRanges(self):
        self.psiRange = [0.0]
        #if k => k+1 ranges. 
        s=0
        for i in xrange(self.nstates):
            s+=self.u0[i]
            self.psiRange.append(s)
        self.psiRange = np.array(self.psiRange)
        assert self.psiRange.shape == (self.nstates+1,)
    def initRangeP(self):
        s=np.zeros(self.nstates)
        self.phiRange = [s]
        for i in xrange(self.nstates):
            s=s+self.P[:,i]
            self.phiRange.append(s)
        self.phiRange = np.array(self.phiRange)
        self.phiRange = self.phiRange.T
        #print self.phiRange
        assert self.phiRange.shape == (self.nstates, self.nstates+1)
    def binarySearch(self, farr, first, second, key):
        mid = (first+second)/2
        #if farr[mid]
    def psi(self, x):
        '''DO NOT CALL DIRECTLY'''
        '''Initialization function'''
        #https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html
        '''There are bugs when x = 0.0 or x = 1.0. I wonder whether there is bug when x is on edge!!! (Fixed)'''
        assert x<=1.0 and x>=0.0, 'x must be in range[0-1]'
        s=np.searchsorted(self.psiRange, x, side='right')
        if s > self.nstates: s-=1
        return s
    def phi(self, si, x):
        '''DO NOT CALL DIRECTLY'''
        '''There are bugs when x = 0.0 or x = 1.0. I wonder whether there is bug when x is on edge!!! (Fixed)'''
        '''Another bug with si (out of bound)'''
        assert x<=1.0 and x>=0.0, 'x must be in range[0-1]'
        s=np.searchsorted(self.phiRange[si-1], x, side='right')
        if s > self.nstates: s-=1
        return s
    def simulate(self, n=10, verbose=False):
        s0 = self.psi(np.random.ranf())
        result = []
        if verbose:
            print 'X0: ', s0
        result.append('X0: %s' % s0)
        prev=s0
        for i in range(1,n):
            x=np.random.ranf()
            curr = self.phi(prev, x)
            result.append('X%s: %s' % (i, curr))
            if verbose:
                print 'X%s' % i, curr, x
            prev = curr
        return result