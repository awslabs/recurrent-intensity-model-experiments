from ..util import extract_past_ij, find_iloc, LazyDenseMatrix
import numpy as np, scipy.sparse as sps
from lightfm import LightFM


class LightFM_BPR:
    def __init__(self, user_rec=False, item_rec=False, epochs=50,
                 no_comp=32, user_alpha=1e-5, item_alpha=1e-5):

        assert user_rec != item_rec, "specify exactly one side to sample negatives"
        self._transposed = user_rec

        self.epochs = epochs
        self.bpr_model = LightFM(
            no_components=no_comp,
            loss='bpr',
            learning_schedule='adagrad',
            user_alpha=user_alpha,
            item_alpha=item_alpha)

    def fit(self, D):
        i, j = extract_past_ij(D.user_df, D.item_df.index)
        train_intn = sps.coo_matrix((np.ones(len(i)), (i, j)),
                                    shape=(len(D.user_df), len(D.item_df))).tocsr()
        if self._transposed:
            train_intn = train_intn.T

        self.bpr_model.fit(train_intn, epochs=self.epochs, verbose=True)
        self.D = D
        return self

    def transform(self, D):
        """ (user_embed * item_embed + user_bias + item_bias).softplus() """

        user_hstack = np.hstack([
            self.bpr_model.user_embeddings,
            self.bpr_model.user_biases[:, None],
            np.ones_like(self.bpr_model.user_biases)[:, None],
        ])

        item_hstack = np.hstack([
            self.bpr_model.item_embeddings,
            np.ones_like(self.bpr_model.item_biases)[:, None],
            self.bpr_model.item_biases[:, None],
        ])

        if self._transposed:
            user_hstack, item_hstack = item_hstack, user_hstack

        user_iloc = find_iloc(self.D.user_df.index, D.user_in_test.index)
        item_iloc = find_iloc(self.D.item_df.index, D.item_in_test.index)
        return (LazyDenseMatrix(user_hstack[user_iloc]) @
                LazyDenseMatrix(item_hstack[item_iloc]).T).softplus()
