import numpy as np
from numpy.core.numeric import indices

class SindyBasisSinTermsGenerator:
    def __init__(self, n_state, n_control) -> None:
        self.n = n_state
        self.m = n_control
    # /__init__()

    def numTerms(self):
        m_plus_n = self.m + self.n
        return m_plus_n * m_plus_n
    #/

    def addToBasis(self, s, u, B, col):
        assert self.n == s.shape[1]
        assert self.m == u.shape[1]
        assert (col + self.numTerms()) <= B.shape[1]

        su = np.concatenate((s,u), axis=1)
        m_plus_n = su.shape[1]
        indices = np.arange(m_plus_n)
        for i in range(m_plus_n):
            B[:, col:col+m_plus_n] = su[:,i][:,np.newaxis] * np.sin(su)
            col += m_plus_n
        #/

        return B, col
    # /addToBasis()

    def extractTerms(self, basis_sublist, sublist_index, basis_offset,
                     s_names, u_names):
        terms = []
        n_basis = len(basis_sublist)

        if sublist_index >= n_basis:
            return terms, sublist_index, basis_offset
        #/
        next_index_to_match = basis_sublist[sublist_index]
        next_index = lambda i, lim: basis_sublist[i] if i < len(basis_sublist) else -1

        m_plus_n = self.m + self.n
        for i in range(m_plus_n):
            first = s_names[i] if i < self.n else u_names[i - self.n]
            for j in range(m_plus_n):
                second = s_names[j] if j < self.n else u_names[j - self.n]
                if basis_offset == next_index_to_match:
                    terms.append(f'{first}*sin({second})')
                    # terms.append(f'{first}[{i}]*sin({second}[{j}])')
                    sublist_index += 1
                    next_index_to_match = next_index(sublist_index, n_basis)
                #/
                basis_offset += 1
            # /for j
        # /for i

        return terms, sublist_index, basis_offset
    # /extractTerms()

# /class SindyBasisSinTermsGenerator
