import torch, dgl, numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from .third_party.lda.lda_model import LatentDirichletAllocation, DocData, WordData, doc_subgraph
from ..util import (empty_cache_on_exit, LowRankDataFrame, create_matrix,
                    _LitValidated, default_random_split, get_batch_size)


class _LitLDA(_LitValidated):
    """ use pytorch lightning to extract random subgraphs """
    def __init__(self, model, G):
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.G = G

        # TODO: this should help checkpoint parameters, but I don't know how
        # pytorch lightning works with distributed parameters.
        for i, w in enumerate(self.model.word_data):
            self.register_buffer(f"word_{i}_nphi", w.nphi)

    def _update_model_from_buffer_change_device(self):
        word_nphi = [getattr(self, f"word_{i}_nphi") for i in range(len(self.model.word_data))]
        self.model.word_data = WordData(self.model.prior['word'], word_nphi)
        self.model.device_list = [nphi.device for nphi in word_nphi]  # cpu

    def forward(self, batch, op=None):
        B = doc_subgraph(self.G, batch.tolist()).to(batch.device)
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

    @empty_cache_on_exit
    def fit(self, D):
        """ learn from training_data on gpu w/ mini-batches; clear gpu in the end """

        user_index = D.user_df[D.user_df['_hist_len'] > 0].index  # prune empty users
        i, j = create_matrix(D.event_df, user_index, self._item_list, 'ij')
        G = dgl.heterograph(
            {('doc', '', 'word'): (i, j)},
            {'doc': len(user_index), 'word': len(self._item_list)},
        )

        lit = _LitLDA(self.model, G)
        trainer = Trainer(
            max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            limit_val_batches=2, callbacks=[lit._checkpoint],
        )
        print(trainer.log_dir)

        train_set, valid_set = default_random_split(np.arange(G.num_nodes('doc')))
        trainer.fit(
            lit,
            DataLoader(train_set, self.batch_size, shuffle=True),
            DataLoader(valid_set, self.batch_size),
        )
        lit._load_best_checkpoint("best")
        lit._update_model_from_buffer_change_device()

        return self

    def transform(self, D, return_doc_data=False):
        """ run e-step to get doc data; output as low-rank nonnegative matrix """

        # create past event df from user_in_test history; _hist_len > 0 avoids na in explode
        past_event_df = D.user_in_test[D.user_in_test['_hist_len'] > 0]['_hist_items'].copy()
        past_event_df.index.name = 'USER_ID'
        past_event_df = past_event_df.explode().to_frame('ITEM_ID').reset_index()

        i, j = create_matrix(past_event_df, D.user_in_test.index, self._item_list, 'ij')
        G = dgl.heterograph(
            {('doc', '', 'word'): (i, j)},
            {'doc': len(D.user_in_test.index), 'word': len(self._item_list)}
        )

        trainer = Trainer(gpus=int(torch.cuda.is_available()))
        doc_nphi = trainer.predict(
            _LitLDA(self.model, G),
            DataLoader(np.arange(G.num_nodes('doc')), self.batch_size),
        )
        doc_data = DocData(self.model.prior['doc'], torch.vstack(doc_nphi))

        u = doc_data._expectation().numpy()
        v = np.vstack([w._expectation().numpy() for w in self.model.word_data])
        out = LowRankDataFrame(
            u, v.T, D.user_in_test.index, np.array(self._item_list), '_nnmf'
        ).reindex(D.item_in_test.index, axis=1, fill_value=0)

        if return_doc_data:
            out = (out, doc_data)
        return out
