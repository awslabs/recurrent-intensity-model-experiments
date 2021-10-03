import torch
from ..util import create_matrix, CustomLowRankDataFrame
from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization

class ALS:
    def __init__(self, factors=32, iterations=50,regularization=0.01, random_state=None,use_native=True,use_cg=True,use_gpu=torch.cuda.is_available()):

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
        df_train = D.event_df[D.event_df['_holdout']==0]
        #create matrix create user-item matrix, whereas implicit takes in item-user matrix
        train_item_user = create_matrix(df_train, D.user_df.index, D.item_df.index, 'csr').T
        self.als_model.fit(train_item_user)
        self.D = D
        return self

    def transform(self, D):
        """ (user_factor * item_factor) """
        assert self.D is D, f"{self.__class__} only transforms training dataset"

        ind_logits = self.als_model.user_factors
        col_logits = self.als_model.item_factors

        return CustomLowRankDataFrame(
            ind_logits, col_logits, 1, D.user_df.index, D.item_df.index, 'raw')


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
            use_gpu=False #gpu version of lmf is not implemented yet
        )

    def fit(self, D):
        df_train = D.event_df[D.event_df['_holdout'] == 0]
        # create matrix create user-item matrix, whereas implicit takes in item-user matrix
        train_item_user = create_matrix(df_train, D.user_df.index, D.item_df.index, 'csr').T
        self.lmf_model.fit(train_item_user)
        self.D = D
        return self

    def transform(self, D):
        """ (user_factor * item_factor) """
        assert self.D is D, f"{self.__class__} only transforms training dataset"

        ind_logits = self.lmf_model.user_factors
        col_logits = self.lmf_model.item_factors

        return CustomLowRankDataFrame(
            ind_logits, col_logits, 1, D.user_df.index, D.item_df.index, 'sigmoid')
