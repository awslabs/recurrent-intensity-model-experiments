import torch, argparse, numpy as np, warnings
from torch.utils.data import DataLoader, random_split
from torch.distributions.categorical import Categorical
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import (_LitValidated, _ReduceLRLoadCkpt,
    empty_cache_on_exit, create_matrix, create_second_order_dataframe)
from .bpr import BPR, _BPR
try:
    import dgl, dgl.function as fn
except ImportError:
    warnings.warn("GCMC requires dgl package")


class _GCMC(_BPR, _LitValidated):
    """ module to compute user RFM embedding. change default lr=0.1 """
    def __init__(self, user_proposal, item_proposal, no_components=32,
        n_negatives=10, lr=0.1, weight_decay=1e-5,
        recency_boundaries=[0.1, 0.3, 1, 3, 10], horizon=float("inf")):
        super(_LitValidated, self).__init__()
        self.register_buffer("user_proposal", torch.as_tensor(user_proposal))
        self.register_buffer("item_proposal", torch.as_tensor(item_proposal))
        self.register_buffer("recency_boundaries",
            torch.as_tensor(recency_boundaries) * horizon)

        self.item_encoder = torch.nn.Embedding(len(item_proposal), no_components)
        self.item_bias_vec = torch.nn.Embedding(len(item_proposal), 1)
        self.conv = dgl.nn.pytorch.conv.GraphConv(no_components, no_components, "none")
        self.recency_encoder = torch.nn.Embedding(len(recency_boundaries)+1, 1)
        self.log_sigmoid = torch.nn.LogSigmoid()

        self.user_rec = True
        self.item_rec = True
        self.n_negatives = n_negatives
        self.lr = lr
        self.weight_decay = weight_decay

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.item_encoder.weight, -initrange, initrange)
        torch.nn.init.zeros_(self.item_bias_vec.weight)
        torch.nn.init.uniform_(self.conv.weight, -initrange, initrange)
        torch.nn.init.zeros_(self.conv.bias)

    def user_encoder(self, i, G=None):
        if G is None:
            G = self.G
        G = G.to(i.device)

        out = self.conv(G, self.item_encoder.weight)
        return out[i]

    def user_bias_vec(self, i, G=None):
        if G is None:
            G = self.G
        G = G.to(i.device)

        G.update_all(lambda x: None, fn.max('t', 'last_t'))
        user_recency = G.nodes['user'].data['test_t'] - G.nodes['user'].data['last_t']
        recency_buckets = torch.bucketize(user_recency, self.recency_boundaries)
        return self.recency_encoder(recency_buckets)[i]


class GCMC:
    def __init__(self, D, batch_size=10000, max_epochs=50, **kw):
        self._user_list = D.training_data.user_df.index.tolist()
        self._padded_item_list = [None] + D.training_data.item_df.index.tolist()

        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self._model_kw = {'horizon': D.horizon}
        self._model_kw.update(kw)

    def _extract_features(self, D):
        """ create item -> user graph """
        D = D.reindex(self._user_list, axis=0).reindex(self._padded_item_list, axis=1)
        D.user_in_test.index.name = 'USER_ID'

        i, j = create_matrix(
            D.user_in_test['_hist_items'].explode().to_frame('ITEM_ID').reset_index(),
            D.user_in_test.index, D.item_in_test.index, 'ij'
            )
        t = np.hstack(D.user_in_test['_timestamps'].apply(lambda x: x[:-1]))

        G = dgl.heterograph(
            {('user','source','item'): (i, j)},
            {'user': len(D.user_in_test.index), 'item': len(D.item_in_test.index)}
        )
        G.edata['t'] = torch.as_tensor(t).double()

        n_users = G.num_nodes('user')
        G = dgl.add_edges(G, range(n_users), [0] * n_users,
            {'t': torch.as_tensor([float("-inf")] * n_users).double()})

        user_test_time = D.user_in_test['_timestamps'].apply(lambda x: x[-1]).values
        G.nodes['user'].data['test_t'] = torch.as_tensor(user_test_time)
        return G.reverse(copy_edata=True)

    @empty_cache_on_exit
    def fit(self, V):
        V = V.reindex(self._user_list, axis=0).reindex(self._padded_item_list, axis=1)
        target_coo = V.target_csr.tocoo()
        dataset = np.transpose([target_coo.row, target_coo.col]).astype(int)

        G = self._extract_features(V)

        user_proposal = (np.ravel(target_coo.sum(axis=1)) + 0.1) ** 0.5
        item_proposal = (np.ravel(target_coo.sum(axis=0)) + 0.1) ** 0.5

        user_incl = (G.in_degrees().numpy() + np.ravel(target_coo.sum(1))).astype(bool)
        item_incl = (G.out_degrees().numpy() + np.ravel(target_coo.sum(0))).astype(bool)

        model = _GCMC(user_proposal * user_incl, item_proposal * item_incl, **self._model_kw)

        N = len(dataset)
        if len(dataset) > 5:
            train_set, valid_set = random_split(dataset, [N*4//5, (N - N*4//5)])
        else:
            train_set = valid_set = dataset

        trainer = Trainer(max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])

        model.G = G
        trainer.fit(model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=(N>1e4)*4),
            DataLoader(valid_set, self.batch_size, num_workers=(N>1e4)*4))
        delattr(model, "G")

        best_model_path = model._checkpoint.best_model_path
        best_model_score = model._checkpoint.best_model_score
        if best_model_score is not None:
            print(f"done fit; best checkpoint {best_model_path} with score {best_model_score}")

        self.item_index = self._padded_item_list
        self.item_embeddings = model.item_encoder.weight.detach().cpu().numpy()
        self.item_biases = model.item_bias_vec.weight.detach().cpu().numpy().ravel()
        self.model = model
        return self

    def transform(self, D):
        G = self._extract_features(D)
        i = torch.arange(G.num_nodes('user'))
        user_embeddings = self.model.user_encoder(i, G).detach().cpu().numpy()
        user_biases = self.model.user_bias_vec(i, G).detach().cpu().numpy().ravel()

        S = create_second_order_dataframe(
            user_embeddings, self.item_embeddings, user_biases, self.item_biases,
            self._user_list, self._padded_item_list, 'sigmoid')

        return S.reindex(D.user_in_test.index, axis=0).reindex(D.item_in_test.index, axis=1)
