import torch, warnings, numpy as np, scipy.sparse as sps
from ..util import extract_past_ij, find_iloc, LazyDenseMatrix
from packaging import version
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization


_to_numpy = lambda x: x.to_numpy() if hasattr(x, 'to_numpy') else x


class ALS:
    def __init__(self, factors=32, iterations=50, regularization=0.01, random_state=None,
                 use_native=True, use_cg=True, use_gpu=torch.cuda.is_available()):

        self.als_model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            random_state=random_state,
            regularization=regularization,
            use_native=use_native,
            use_cg=use_cg,
            use_gpu=use_gpu
        )

    def fit(self, D):
        i, j = extract_past_ij(D.user_df, D.item_df.index)
        train_user_item = sps.coo_matrix((np.ones(len(i)), (i, j)),
                                         shape=(len(D.user_df), len(D.item_df))).tocsr()
        if version.parse(implicit.__version__) >= version.parse('0.5'):
            self.als_model.fit(train_user_item)
        else:
            self.als_model.fit(train_user_item.T)
        self.user_hstack = _to_numpy(self.als_model.user_factors)
        self.item_hstack = _to_numpy(self.als_model.item_factors)
        delattr(self, "als_model")

        self.D = D
        return self

    def transform(self, D):
        """ (user_factor * item_factor) """
        user_iloc = find_iloc(self.D.user_df.index, D.user_in_test.index)
        item_iloc = find_iloc(self.D.item_df.index, D.item_in_test.index)
        return (LazyDenseMatrix(self.user_hstack[user_iloc]) @
                LazyDenseMatrix(self.item_hstack[item_iloc]).T).softplus()


class LogisticMF:
    def __init__(self, factors=30, iterations=30, neg_prop=30, learning_rate=1.00, regularization=0.6,
                 random_state=None):
        self.lmf_model = LogisticMatrixFactorization(
            factors=factors,
            learning_rate=learning_rate,
            iterations=iterations,
            regularization=regularization,
            neg_prop=neg_prop,
            random_state=random_state,
            use_gpu=False  # gpu version of lmf is not implemented yet
        )

    def fit(self, D):
        i, j = extract_past_ij(D.user_df, D.item_df.index)
        train_user_item = sps.coo_matrix((np.ones(len(i)), (i, j)),
                                         shape=(len(D.user_df), len(D.item_df))).tocsr()
        if version.parse(implicit.__version__) >= version.parse('0.5'):
            self.lmf_model.fit(train_user_item)
        else:
            self.lmf_model.fit(train_user_item.T)
        self.D = D
        return self

    def transform(self, D):
        """ (user_factor * item_factor) """

        user_hstack = _to_numpy(self.lmf_model.user_factors)
        item_hstack = _to_numpy(self.lmf_model.item_factors)

        user_iloc = find_iloc(self.D.user_df.index, D.user_in_test.index)
        item_iloc = find_iloc(self.D.item_df.index, D.item_in_test.index)
        return (LazyDenseMatrix(user_hstack[user_iloc]) @
                LazyDenseMatrix(item_hstack[item_iloc]).T).sigmoid()
