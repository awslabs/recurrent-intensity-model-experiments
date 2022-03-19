import torch, dgl, numpy as np, pandas as pd
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from .third_party.lda.lda_model import LatentDirichletAllocation, DocData, WordData
from ..util import (empty_cache_on_exit, find_iloc, LazyDenseMatrix, indices2csr,
                    _LitValidated, default_random_split, get_batch_size)


class _LitLDA(_LitValidated):
    """ use pytorch lightning to extract random subgraphs """
    def __init__(self, model):
        super().__init__()
        self.automatic_optimization = False
        self.model = model

        # TODO: this should help checkpoint parameters, but I don't know how
        # pytorch lightning works with distributed parameters.
        for i, w in enumerate(self.model.word_data):
            self.register_buffer(f"word_{i}_nphi", w.nphi)

    def _update_model_from_buffer_change_device(self):
        word_nphi = [getattr(self, f"word_{i}_nphi") for i in range(len(self.model.word_data))]
        self.model.word_data = WordData(self.model.prior['word'], word_nphi)
        self.model.device_list = [nphi.device for nphi in word_nphi]  # cpu

    def forward(self, batch, op=None):
        i = torch.hstack([torch.ones_like(j) * i for i, j in enumerate(batch)])
        j = torch.hstack(batch)
        unique_j, inverse_j = torch.unique(j, return_inverse=True)
        B = dgl.heterograph({('doc', '', 'word'): (i, inverse_j)})
        B.nodes['word'].data['_ID'] = unique_j
        if op is None:
            op = lambda B: self.model.transform(B).nphi.to("cpu")
        return op(B)

    @torch.no_grad()
    def training_step(self, batch, batch_idx):
        self.model.verbose = (batch_idx == 0)
        self.forward(batch, self.model.partial_fit)

    def validation_step(self, batch, batch_idx):
        ppl = self.forward(batch, self.model.perplexity)
        return torch.as_tensor(ppl).log()

    def configure_optimizers(self):
        pass


class LDA:
    """ run lda example in dgl codebase; use gpus and mini-batches for scalability """
    def __init__(
        self, D, n_components=128, batch_size=None, rho=None, max_epochs=20,
        mult={'doc': 1, 'word': 1000}, **kw
    ):
        self._item_list = D.item_df.index.tolist()
        nnz_users = (D.user_df['_hist_len'] > 0).sum()

        if batch_size is None:
            batch_size = get_batch_size((nnz_users, len(self._item_list))) * 50  # sparse graph
        self.batch_size = batch_size

        if rho is None:
            rho = np.clip(batch_size / nnz_users, 0.01, 1)
        self.model = LatentDirichletAllocation(
            len(self._item_list), n_components, mult=mult, rho=rho, **kw)

        self.max_epochs = max_epochs
        print(f"LDA initiated, batch_size={batch_size}, rho={rho}")

    def _create_past_tokens(self, user_df):
        """ create doc->word graph, including empty docs """
        tokenize = {k: j for j, k in enumerate(self._item_list)}
        past_tokens = user_df['_hist_items'].apply(
            lambda x: [tokenize[y] for y in x if y in tokenize])
        return past_tokens.tolist()

    @empty_cache_on_exit
    def fit(self, D):
        """ learn from D.user_df on gpu w/ mini-batches; clear gpu in the end """

        past_tokens = self._create_past_tokens(D.user_df[D.user_df['_hist_len'] > 0])
        lit = _LitLDA(self.model)
        trainer = Trainer(
            max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            limit_val_batches=2, callbacks=[lit._checkpoint],
        )
        print(trainer.log_dir)

        train_set, valid_set = default_random_split(past_tokens)
        trainer.fit(
            lit,
            DataLoader(train_set, self.batch_size, shuffle=True, collate_fn=_collate_fn),
            DataLoader(valid_set, self.batch_size, collate_fn=_collate_fn),
        )
        lit._load_best_checkpoint("best")
        lit._update_model_from_buffer_change_device()

        return self

    def transform(self, D, return_doc_data=False):
        """ run e-step to get doc data; output as low-rank nonnegative matrix """

        past_tokens = self._create_past_tokens(D.user_in_test)
        trainer = Trainer(gpus=int(torch.cuda.is_available()))
        doc_nphi = trainer.predict(
            _LitLDA(self.model),
            DataLoader(past_tokens, self.batch_size, collate_fn=_collate_fn),
        )
        doc_data = DocData(self.model.prior['doc'], torch.vstack(doc_nphi))

        u = doc_data._expectation().numpy()
        vT = np.vstack([w._expectation().numpy() for w in self.model.word_data])
        v_iloc = find_iloc(self._item_list, D.item_in_test.index)
        out = LazyDenseMatrix(u) @ LazyDenseMatrix(vT[:, v_iloc])

        if return_doc_data:
            out = (out, doc_data)
        return out


def _collate_fn(x):
    """ works for empty lists; pytorch-lightning somehow yields the right device as well """
    return [torch.as_tensor(y, dtype=torch.int64) for y in x]
