from ..util import create_matrix, ExponentiatedLowRankDataFrame
import numpy as np, pandas as pd
from lightfm import LightFM


class LightFM_BPR:
    def __init__(self, user_rec=False, item_rec=False, epochs=50,
        no_comp=32, user_alpha=1e-5, item_alpha=1e-5):

        assert user_rec != item_rec, "specify exactly one side to sample negatives"
        self.user_rec = user_rec

        self.epochs=epochs
        self.bpr_model = LightFM(
            no_components=no_comp,
            loss='bpr',
            learning_schedule='adagrad',
            user_alpha=user_alpha,
            item_alpha=item_alpha)

    def fit(self, D):
        df_train = D.event_df[D.event_df['_holdout']==0]
        train_intn = create_matrix(df_train, D.user_df.index, D.item_df.index, 'csr')
        if self.user_rec:
            train_intn = train_intn.T

        self.bpr_model.fit(train_intn, epochs=self.epochs, verbose=True)
        self.D = D
        return self

    def transform(self, D):
        """ (user_embed * item_embed + user_bias + item_bias).exp() """
        assert self.D is D, f"{self.__class__} only transforms training dataset"

        ind_logits = np.hstack([
            self.bpr_model.user_embeddings,
            self.bpr_model.user_biases[:, None],
            np.ones_like(self.bpr_model.user_biases)[:, None],
            ])

        col_logits = np.hstack([
            self.bpr_model.item_embeddings,
            np.ones_like(self.bpr_model.item_biases)[:, None],
            self.bpr_model.item_biases[:, None],
            ])

        if self.user_rec:
            ind_logits, col_logits = col_logits, ind_logits

        return ExponentiatedLowRankDataFrame(
            ind_logits, col_logits, 1, D.user_df.index, D.item_df.index)
