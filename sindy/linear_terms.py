import numpy as np

class SindyBasisLinearTermsGenerator:
    def __init__(self, n_state, n_control) -> None:
        self.n = n_state
        self.m = n_control
    # /__init__()

    def numTerms(self):
        return self.n + self.m
    #/

    def addToBasis(self, s, u, B, col):
        assert self.n == s.shape[1]
        assert self.m == u.shape[1]
        assert (col + self.numTerms()) <= B.shape[1]

        B[:, col:col+self.n] = s
        col += self.n
        
        B[:, col : col+self.m] = u
        col += self.m

        return B, col
    # /addToBasis()

    def extractTerms(self, basis_sublist, sublist_index, basis_offset):
        terms = []
        assert len(basis_sublist) > 0
        assert basis_offset == 1 # must be the second check

        # State terms
        while sublist_index < len(basis_sublist):
            i = basis_sublist[sublist_index] - basis_offset
            if i < self.n:
                terms.append(f's[{i}]')
            else:
                break
            # /if-else
            sublist_index += 1
        # /while
        basis_offset += self.n

        # Control terms
        while sublist_index < len(basis_sublist):
            i = basis_sublist[sublist_index] - basis_offset
            if i < self.m:
                terms.append(f'u[{i}]')
            else:
                break
            # /if-else
            sublist_index += 1
        # /while
        basis_offset += self.m

        return terms, sublist_index, basis_offset
    # /extractTerms()

# /class SindyBasisLinearTermsGenerator
