import pandas as pd, numpy as np, torch, scipy.sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from rime.util.score_array import auto_cast_lazy_score
from collections.abc import Iterable


class TF_IDF:
    """ create cosine similarity between the last user item and the item to recommend based on tf-idf embedding """
    def __init__(self, item_df):
        assert "TITLE" in item_df or "embedding" in item_df, "require TITLE or embedding"

        self.item_id = item_df.index
        if 'embedding' in item_df:
            self.tfidf_csr = np.vstack(item_df['embedding'].tolist())
            zeros = np.zeros_like(self.tfidf_csr[:1])
            self.tfidf_pad_zeros = np.vstack([self.tfidf_csr, zeros])
        else:
            self.tfidf_fit = TfidfVectorizer().fit(item_df['TITLE'].tolist())
            self.tfidf_csr = self.tfidf_fit.transform(item_df['TITLE'].tolist())
            zeros = self.tfidf_csr[:1] * 0
            zeros.eliminate_zeros()
            self.tfidf_pad_zeros = sps.vstack([self.tfidf_csr, zeros])

    def fit(self, *args, **kw):
        return self

    def transform(self, D):
        user_last_item = D.user_in_test['_hist_items'].apply(
            lambda x: x[-1] if isinstance(x, Iterable) else None)
        user_last_item_index = self.item_id.get_indexer(user_last_item.values)
        user_emb = self.tfidf_pad_zeros[user_last_item_index]

        item_index = self.item_id.get_indexer(D.item_in_test.index)
        item_emb = self.tfidf_csr[item_index]
        return auto_cast_lazy_score(user_emb) @ auto_cast_lazy_score(item_emb).T
