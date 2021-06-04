
class SindyBasisConstantTermGenerator:
    def __init__(self) -> None:
        pass
    # /__init__()

    def numTerms(self):
        return 1
    #/

    def addToBasis(self, s, u, B, col):
        assert col == 0 # must be first to be called
        assert (col + self.numTerms()) <= B.shape[1]
        B[:,0].fill(1)
        col = 1
        return B, col
    # /addToBasis()

    def extractTerms(self, basis_sublist, sublist_index, basis_offset,
                     s_names, u_names):
        terms = []
        assert len(basis_sublist) > 0
        assert sublist_index == 0 # can only be first item in list
        assert basis_offset == 0 # must be first check
        if basis_sublist[sublist_index] == 0:
            terms = ['1']
            sublist_index = 1
        #/
        basis_offset = 1
        return terms, sublist_index, basis_offset
    # /extractTerms()

# /class SindyBasisConstantTermGenerator
