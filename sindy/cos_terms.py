import numpy as np

class SindyBasisCosTermsGenerator:
    def __init__(self, n_state, n_control) -> None:
        self.n = n_state
        self.m = n_control
    # /__init__()

    def numTerms(self):
        return 2 * self.m * self.n
    #/

    def addToBasis(self, s, u, B, col):
        assert self.n == s.shape[1]
        assert self.m == u.shape[1]
        assert (col + self.numTerms()) <= B.shape[1]

        # s_i cos(u_j) terms
        for i in range(self.n):
            B[:, col:col+self.m] = s[:,i][:,np.newaxis] * np.cos(u)
            col += self.m
        # /for i

        # u_i cos(s_j) terms
        for i in range(self.m):
            B[:, col:col+self.n] = u[:,i][:,np.newaxis] * np.cos(s)
            col += self.n
        # /for i

        return B, col
    # /addToBasis()

    def extractTerms(self, basis_sublist, sublist_index, basis_offset):
        terms = []
        n_basis = len(basis_sublist)
        next_index_to_match = basis_sublist[sublist_index]
        
        next_index = lambda i, lim: basis_sublist[i] if i < len(basis_sublist) else -1

        for i in range(self.n):
            for j in range(self.m):
                if basis_offset == next_index_to_match:
                    terms.append(f's[{i}]*cos(u[{j}])')
                    sublist_index += 1
                    next_index_to_match = next_index(sublist_index, n_basis)
                #/
                basis_offset += 1
            # /for j
        # /for i

        for i in range(self.m):
            for j in range(self.n):
                if basis_offset == next_index_to_match:
                    terms.append(f's[{j}]*cos(u[{i}])')
                    sublist_index += 1
                    next_index_to_match = next_index(sublist_index, n_basis)
                #/
                basis_offset += 1
            # /for j
        # /for i

        return terms, sublist_index, basis_offset
    # /extractTerms()

# /class SindyBasisCosTermsGenerator
