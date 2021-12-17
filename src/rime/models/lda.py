import torch, dgl, numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from .third_party.lda.lda_model import LatentDirichletAllocation, WordData, doc_subgraph
from ..util import (empty_cache_on_exit, LowRankDataFrame, create_matrix,
                    _LitValidated, default_train_valid_loaders, get_batch_size)


class _LitLDA(_LitValidated):
    """ use pytorch lightning to extract random subgraphs """
    def __init__(self, model, G):
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.G = G

    def forward(self, batch, op=None):
        B = doc_subgraph(self.G, batch.tolist()).to(batch.device)
        if op is None:
            op = lambda B: self.model.transform(B)._expectation().to('cpu').numpy()
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
        mult={'doc': 1, 'word': 100}, **kw
    ):
        self._item_list = D.item_df.index.tolist()

        if batch_size is None:
            batch_size = get_batch_size((len(D.user_df), len(D.item_df))) * 20  # sparse graph
        self.batch_size = batch_size

        if rho is None:
            rho = np.clip(batch_size / len(D.user_df), 0.01, 1)
        self.model = LatentDirichletAllocation(
            len(self._item_list), n_components, mult=mult, rho=rho, **kw)

        self.max_epochs = max_epochs

    @empty_cache_on_exit
    def fit(self, D):
        """ learn from training_data on gpu w/ mini-batches; clear gpu in the end """

        i, j = create_matrix(D.event_df, D.user_df.index, self._item_list, 'ij')
        G = dgl.heterograph(
            {('doc', '', 'word'): (i, j)},
            {'doc': len(D.user_df.index), 'word': len(self._item_list)},
        )

        lit = _LitLDA(self.model, G)
        trainer = Trainer(
            max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            limit_val_batches=2
        )
        print(trainer.log_dir)
        trainer.fit(lit, *default_train_valid_loaders(
            np.arange(G.num_nodes('doc')), batch_size=self.batch_size
        ))
        self.model.to('cpu')
        return self

    def transform(self, D):
        """ run e-step to get doc data; output as low-rank nonnegative matrix """

        # create matrix from user_in_test history; avoid na in explode
        user_in_test_hist = D.user_in_test[D.user_in_test['_hist_len'] > 0]['_hist_items'].copy()
        user_in_test_hist.index.name = 'USER_ID'
        i, j = create_matrix(
            user_in_test_hist.explode().to_frame('ITEM_ID').reset_index(),
            D.user_in_test.index, self._item_list, 'ij'
        )
        G = dgl.heterograph(
            {('doc', '', 'word'): (i, j)},
            {'doc': len(D.user_in_test.index), 'word': len(self._item_list)}
        )

        trainer = Trainer(gpus=int(torch.cuda.is_available()))
        doc_exp = trainer.predict(
            _LitLDA(self.model, G),
            DataLoader(np.arange(G.num_nodes('doc')), batch_size=self.batch_size),
        )
        u = np.vstack(doc_exp)
        v = np.vstack([w._expectation().numpy() for w in self.model.word_data])
        return LowRankDataFrame(
            u, v.T, D.user_in_test.index, np.array(self._item_list), '_nnmf'
        ).reindex(D.item_in_test.index, axis=1, fill_value=0)
