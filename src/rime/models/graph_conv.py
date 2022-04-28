import torch, numpy as np, warnings, pandas as pd, collections, torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import (_LitValidated, empty_cache_on_exit, LazyDenseMatrix, matrix_reindex,
                    default_random_split, auto_cast_lazy_score)
from .bpr import _BPR_Common
import dgl, dgl.function as fn


def _plain_average(G, item_embeddings):
    with G.local_scope():
        G.srcdata['h'] = item_embeddings
        G.update_all(fn.copy_src('h', 'm'), fn.mean(msg='m', out='h'))
        return G.dstdata['h']


class _GraphConv(_BPR_Common):
    """ module to compute user RFM embedding.
    """
    def __init__(self, n_users=None, n_items=None,
                 user_rec=True, item_rec=True, no_components=32,
                 n_negatives=10, lr=1, weight_decay=1e-5, training_prior_fcn=lambda x: x,
                 user_conv_model='GCN',  # plain_average
                 user_embeddings=None, item_embeddings=None, item_zero_bias=False,
                 recency_boundary_multipliers=[0.1, 0.3, 1, 3, 10], horizon=float("inf"),
                 user_drop=0, item_drop=0):

        super().__init__(user_rec, item_rec, n_negatives, lr, weight_decay,
                         training_prior_fcn)

        if item_embeddings is not None:
            warnings.warn("setting no_components according to provided embeddings")
            no_components = item_embeddings.shape[-1]

        self.item_encoder = torch.nn.Embedding(n_items, no_components)
        if item_embeddings is not None:
            self.item_encoder.weight.requires_grad = False
            self.item_encoder.weight.copy_(torch.as_tensor(item_embeddings))

        self.item_bias_vec = torch.nn.Embedding(n_items, 1)
        if item_zero_bias:
            self.item_bias_vec.weight.requires_grad = False
            self.item_bias_vec.weight.copy_(torch.zeros_like(self.item_bias_vec.weight))

        if user_conv_model == 'GCN':
            self.user_conv = dgl.nn.pytorch.conv.GraphConv(no_components, no_components, "none")
        elif user_conv_model == 'plain_average':
            self.user_conv = _plain_average
        self.user_layer_norm = torch.nn.LayerNorm(no_components)
        if user_embeddings is not None:
            self.user_ext_layer_norm = torch.nn.LayerNorm(user_embeddings.shape[1])

        self.register_buffer("recency_boundaries",
                             torch.as_tensor(recency_boundary_multipliers) * horizon)
        self.recency_encoder = torch.nn.Embedding(len(recency_boundary_multipliers) + 1, 1)
        self.user_drop = torch.nn.Dropout(user_drop)
        self.item_drop = torch.nn.Dropout(item_drop)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.item_encoder.weight.requires_grad:
            torch.nn.init.uniform_(self.item_encoder.weight, -initrange, initrange)
        if self.item_bias_vec.weight.requires_grad:
            torch.nn.init.zeros_(self.item_bias_vec.weight)
        if hasattr(self.user_conv, "weight"):
            torch.nn.init.uniform_(self.user_conv.weight, -initrange, initrange)
        if hasattr(self.user_conv, "bias"):
            torch.nn.init.zeros_(self.user_conv.bias)

    def _user_subgraph(self, i, G):
        I, i_reverse = torch.unique(i, return_inverse=True)
        if len(I) < 0.5 * G.num_nodes('user'):
            sampler = dgl.dataloading.neighbor.MultiLayerFullNeighborSampler(1)
            mfg = sampler.sample_blocks(G.to(I.device), {'user': I})[0]
            sub_G = dgl.edge_type_subgraph(dgl.block_to_graph(mfg), mfg.etypes)
            return i_reverse, sub_G, sub_G.srcdata['_ID']
        else:
            return i, G, torch.arange(G.num_nodes('item')).to(i)

    def user_encoder(self, i, G):
        i, G, src_j = self._user_subgraph(i, G)
        item_embeddings = self.item_encoder(src_j)
        user_embeddings = self.user_layer_norm(self.user_conv(G, item_embeddings))
        if 'embedding' in G.dstdata:
            user_ext = self.user_ext_layer_norm(G.dstdata['embedding'])
            user0, user1 = torch.split(user_embeddings, [
                user_embeddings.shape[1] - user_ext.shape[1], user_ext.shape[1]], 1)
            user_embeddings = torch.cat([user0, user1 + user_ext], 1)
        return user_embeddings[i]

    def user_bias_vec(self, i, G):
        with G.local_scope():
            G.update_all(lambda x: None, fn.max('t', 'last_t'))
            user_recency = G.nodes['user'].data['test_t'] - G.nodes['user'].data['last_t']
        recency_buckets = torch.bucketize(user_recency, self.recency_boundaries)
        return self.recency_encoder(recency_buckets)[i]

    def on_fit_start(self):
        if hasattr(self, "prior_score") and not hasattr(self, "prior_score_T"):
            self.prior_score_T = [getattr(p, "T", None) for p in self.prior_score]
        self.G_list = [G.to(self.device) for G in self.G_list]

    def training_step(self, batch, batch_idx):
        k = batch[:, -1]
        loss_list = []
        for s, (G, u_p, i_p, p, pT) in enumerate(zip(
            self.G_list, self.user_proposal, self.item_proposal,
            self.prior_score, self.prior_score_T
        )):
            if any(k == s):
                single_loss = self._bpr_training_step(batch[k == s, :-1], u_p, i_p, p, pT,
                                                      user_kw={"G": G})
                loss_list.append(single_loss)

        loss = torch.stack(loss_list).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def forward(self, i, j, user_kw={}):
        user_embeddings = self.user_drop(self.user_encoder(i, **user_kw))
        item_embeddings = self.item_drop(self.item_encoder(j))

        return (user_embeddings * item_embeddings).sum(-1) \
            + self.user_drop(self.user_bias_vec(i, **user_kw)).squeeze(-1) \
            + self.item_drop(self.item_bias_vec(j)).squeeze(-1)


class GraphConv:
    def __init__(self, D, batch_size=10000, max_epochs=50,
                 sample_with_prior=True, sample_with_posterior=0.5,
                 truncated_input_steps=256, auto_pad_item=True, **kw):
        self._padded_item_list = [None] * auto_pad_item + D.item_df.index.tolist()
        self._tokenize = {k: j for j, k in enumerate(self._padded_item_list)}
        self.n_tokens = len(self._tokenize)
        self._truncated_input_steps = truncated_input_steps

        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.sample_with_prior = sample_with_prior
        self.sample_with_posterior = sample_with_posterior

        self._model_kw = {'horizon': D.horizon}
        if "embedding" in D.item_df:
            item_embeddings = np.vstack(D.item_df["embedding"]).astype('float32')
            item_embeddings = np.pad(item_embeddings, ((1, 0), (0, 0)), constant_values=0)
            self._model_kw["item_embeddings"] = item_embeddings
        self._model_kw.update(kw)

    def _extract_features(self, D):
        """ create item -> user graph; allow same USER_ID with different TEST_START_TIME """

        user_non_empty = D.user_in_test.reset_index()[D.user_in_test['_hist_len'].values > 0]

        past_event_df = user_non_empty['_hist_items'].apply(
            lambda x: x[-self._truncated_input_steps:]
        ).explode().to_frame("ITEM_ID").assign(
            TIMESTAMP=user_non_empty['_hist_ts'].apply(
                lambda x: x[-self._truncated_input_steps:]).explode().values
        ).assign(j=lambda x: x['ITEM_ID'].apply(lambda k: self._tokenize.get(k, -1)))
        past_event_df = past_event_df[past_event_df['j'] > -1]  # ignore oov items

        G = dgl.heterograph(
            {('user', 'source', 'item'): (past_event_df.index.values,
                                          past_event_df["j"].values)},
            {'user': len(D.user_in_test), 'item': self.n_tokens}
        )
        G.edata['t'] = torch.as_tensor(past_event_df["TIMESTAMP"].values.astype('float64'))

        # add padding item to guard against users with empty histories
        n_users = G.num_nodes('user')
        pad_time = -np.inf * torch.ones(n_users).double()
        G = dgl.add_edges(G, range(n_users), [0] * n_users, {'t': pad_time})

        user_test_time = D.user_in_test['TEST_START_TIME'].values
        G.nodes['user'].data['test_t'] = torch.as_tensor(user_test_time)

        if hasattr(D.user_in_test, 'embedding'):
            user_ext_embeddings = np.vstack(D.user_in_test['embedding']).astype('float32')
            G.nodes['user'].data['embedding'] = torch.as_tensor(user_ext_embeddings)
        return G.reverse(copy_edata=True)

    def _extract_task(self, k, V):
        user_proposal = np.ravel(V.target_csr.sum(axis=1) + 0.1) ** self.sample_with_posterior
        item_proposal = np.ravel(V.target_csr.sum(axis=0) + 0.1) ** self.sample_with_posterior

        item_proposal = pd.Series(item_proposal, V.item_in_test.index) \
                        .reindex(self._padded_item_list, fill_value=0).values

        V = V.reindex(self._padded_item_list, axis=1)
        target_coo = V.target_csr.tocoo()
        dataset = np.array([
            (i, j, w, k) for i, j, w in zip(target_coo.row, target_coo.col, target_coo.data)
        ])
        G = self._extract_features(V)
        return dataset, G, user_proposal, item_proposal, getattr(V, "prior_score", None)

    @empty_cache_on_exit
    def fit(self, *V_arr):
        V_arr = [V for V in V_arr if V is not None]
        if len(V_arr) == 0:
            self.model = _GraphConv(None, self.n_tokens, **self._model_kw)
            return self

        dataset, G_list, user_proposal, item_proposal, prior_score = zip(*[
            self._extract_task(k, V) for k, V in enumerate(V_arr)
        ])

        print("GraphConv label sizes", [len(d) for d in dataset])
        dataset = np.vstack(dataset)

        if "embedding" in V_arr[0].user_in_test:
            self._model_kw["user_embeddings"] = np.vstack(
                V_arr[0].user_in_test['embedding'].iloc[:1])  # just need shape[1]
        model = _GraphConv(None, self.n_tokens, **self._model_kw)

        N = len(dataset)
        train_set, valid_set = default_random_split(dataset)

        trainer = Trainer(
            max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])

        model.G_list = G_list
        model.user_proposal = user_proposal
        model.item_proposal = item_proposal
        if self.sample_with_prior:
            model.prior_score = [auto_cast_lazy_score(p) for p in prior_score]
        else:
            model.prior_score = [None for p in prior_score]

        valid_batch_size = self.batch_size * model.n_negatives * 2 // model.valid_n_negatives
        trainer.fit(
            model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=(N > 1e4) * 4),
            DataLoader(valid_set, valid_batch_size, num_workers=(N > 1e4) * 4))
        model._load_best_checkpoint("best")
        for attr in ['G_list', 'user_proposal', 'item_proposal', 'prior_score', 'prior_score_T']:
            delattr(model, attr)
        self.model = model
        return self

    @property
    def item_embeddings(self):
        return self.model.item_encoder(torch.arange(self.n_tokens)).detach().cpu().numpy()

    @property
    def item_biases(self):
        return self.model.item_bias_vec(torch.arange(self.n_tokens)).detach().cpu().numpy().ravel()

    def transform(self, D):
        G = self._extract_features(D)
        i = torch.arange(G.num_nodes('user'))
        self.model.to("cpu")
        user_embeddings = self.model.user_encoder(i, G).detach().cpu().numpy()
        user_biases = self.model.user_bias_vec(i, G).detach().cpu().numpy().ravel()

        item_reindex = lambda x, fill_value=0: matrix_reindex(
            x, self._padded_item_list, D.item_in_test.index, axis=0, fill_value=fill_value)
        item_embeddings = item_reindex(self.item_embeddings)
        item_biases = item_reindex(self.item_biases, fill_value=-np.inf)
        return (LazyDenseMatrix(user_embeddings).apply(self.model.user_drop) @
                LazyDenseMatrix(item_embeddings).apply(self.model.item_drop).T
                + LazyDenseMatrix(user_biases[:, None]).apply(self.model.user_drop)
                + LazyDenseMatrix(item_biases[None, :]).apply(self.model.item_drop)
                ).softplus()
