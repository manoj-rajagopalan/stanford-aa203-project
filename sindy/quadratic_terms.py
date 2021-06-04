import numpy as np

class SindyBasisQuadraticTermsGenerator:
    def __init__(self, n_state, n_control) -> None:
        self.n = n_state
        self.m = n_control
    # /__init__()

    def numTerms(self):
        return self.n * (self.n+1)//2 + self.m * self.n + self.m * (self.m+1)//2
    #/

    def addToBasis(self, s, u, B, col):
        assert self.n == s.shape[1]
        assert self.m == u.shape[1]
        assert (col + self.numTerms()) <= B.shape[1]
        n_sqr = self.n * self.n
        m_sqr = self.m * self.m
        mn = self.m * self.n

        # quadratic state terms
        for i in range(self.n):
            B[:, col:col+self.n-i] = s[:,i][:,np.newaxis] * s[:, i:]
            col += self.n - i
        # /for i

        # quadratic control terms
        for j in range(self.m):
            B[:, col : col + self.m - j] = u[:,j][:, np.newaxis] * u[:, j:]
            col += self.m - j
        # /for j

        # quadratic state-control terms
        for i in range(self.n):
            B[:, col : col + self.m] = s[:,i][:, np.newaxis] * u
            col += self.m
        # /for i

        return B, col
    # /addToBasis()

    def extractTerms(self, basis_sublist, sublist_index, basis_offset):
        terms = []
        n_basis = len(basis_sublist)
        next_index_to_match = basis_sublist[sublist_index]
        
        next_index = lambda i, lim: basis_sublist[i] if i < len(basis_sublist) else -1

        # quadratic state terms
        for i in range(self.n):
            for j in range(i, self.n):
                if basis_offset == next_index_to_match:
                    terms.append(f's[{i}]*s[{j}]')
                    sublist_index += 1
                    next_index_to_match = next_index(sublist_index, n_basis)
                #/ if
                basis_offset += 1
            # /for j
        # /for i

        # quadratic control terms
        for i in range(self.m):
            for j in range(i, self.m):
                if basis_offset == next_index_to_match:
                    terms.append(f'u[{i}]*u[{j}]')
                    sublist_index += 1
                    next_index_to_match = next_index(sublist_index, n_basis)
                #/ if
                basis_offset += 1
            # /for j
        # /for i

        # quadratic state-control terms
        for i in range(self.n):
            for j in range(self.m):
                if basis_offset == next_index_to_match:
                    terms.append(f's[{i}]*u[{j}]')
                    sublist_index += 1
                    next_index_to_match = next_index(sublist_index, n_basis)
                #/ if
                basis_offset += 1
            # /for j
        # /for i

        return terms, sublist_index, basis_offset
    # /extractTerms()

# /class SindyBasisQuadraticTermsGenerator
