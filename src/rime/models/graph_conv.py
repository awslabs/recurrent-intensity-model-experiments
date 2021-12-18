import torch, numpy as np, warnings, pandas as pd
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import (_LitValidated, empty_cache_on_exit, create_matrix,
                    create_second_order_dataframe)
from .bpr import BPR, _BPR
try:
    import dgl, dgl.function as fn
except ImportError:
    warnings.warn("GraphConv requires dgl package")


class _GraphConv(_BPR, _LitValidated):
    """ module to compute user RFM embedding.
    """
    def __init__(self, *args, no_components=32, encode_user_ids="n/a",
                 recency_multipliers=[0.1, 0.3, 1, 3, 10], horizon=float("inf"),
                 **kw):

        super().__init__(*args,
                         no_components=no_components, encode_user_ids=False,
                         **kw)

        self.register_buffer("recency_boundaries",
                             torch.as_tensor(recency_multipliers) * horizon)

        self.conv = dgl.nn.pytorch.conv.GraphConv(no_components, no_components, "none")
        self.layer_norm = torch.nn.LayerNorm(no_components)
        self.recency_encoder = torch.nn.Embedding(len(self.recency_boundaries) + 1, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.item_encoder.weight, -initrange, initrange)
        torch.nn.init.zeros_(self.item_bias_vec.weight)
        torch.nn.init.uniform_(self.conv.weight, -initrange, initrange)
        torch.nn.init.zeros_(self.conv.bias)

    def user_encoder(self, i, G):
        G = G.to(i.device)
        out = self.conv(G, self.item_encoder.weight)
        return self.layer_norm(out)[i]

    def user_bias_vec(self, i, G):
        G = G.to(i.device).clone()
        G.update_all(lambda x: None, fn.max('t', 'last_t'))
        user_recency = G.nodes['user'].data['test_t'] - G.nodes['user'].data['last_t']
        recency_buckets = torch.bucketize(user_recency, self.recency_boundaries)
        return self.recency_encoder(recency_buckets)[i]

    def forward(self, i, j, G):
        return (self.user_encoder(i, G) * self.item_encoder(j)).sum(-1) \
            + self.user_bias_vec(i, G).squeeze(-1) + self.item_bias_vec(j).squeeze(-1)

    def training_step(self, batch, batch_idx):
        k = batch[:, -1]
        loss_list = []
        for s, (G, u_p, i_p) in enumerate(zip(
            self.G_list, self.user_proposal, self.item_proposal
        )):
            single_loss = self._bpr_training_step(batch[k == s, :-1], u_p, i_p, G=G)
            loss_list.append(single_loss)

        loss = torch.stack(loss_list).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss


class GraphConv:
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

        # create past event df from user_in_test history; _hist_len > 0 avoids na in explode
        past_event_df = D.user_in_test[D.user_in_test['_hist_len'] > 0]['_hist_items'].copy()
        past_event_df.index.name = 'USER_ID'
        past_event_df = past_event_df.explode().to_frame('ITEM_ID').reset_index()

        i, j = create_matrix(past_event_df, D.user_in_test.index, D.item_in_test.index, 'ij')
        t = np.hstack(D.user_in_test['_timestamps'].apply(lambda x: x[:-1]))  # empty is okay

        G = dgl.heterograph(
            {('user', 'source', 'item'): (i, j)},
            {'user': len(D.user_in_test.index), 'item': len(D.item_in_test.index)}
        )
        G.edata['t'] = torch.as_tensor(t).double()

        # add padding item to avoid empty users
        n_users = G.num_nodes('user')
        pad_time = -np.inf * torch.ones(n_users).double()
        G = dgl.add_edges(G, range(n_users), [0] * n_users, {'t': pad_time})

        user_test_time = D.user_in_test['_timestamps'].apply(lambda x: x[-1]).values
        G.nodes['user'].data['test_t'] = torch.as_tensor(user_test_time)
        return G.reverse(copy_edata=True)

    def _extract_task(self, k, V):
        user_proposal = np.ravel(V.target_csr.sum(axis=1) + 0.1) ** 0.5
        item_proposal = np.ravel(V.target_csr.sum(axis=0) + 0.1) ** 0.5

        user_proposal = pd.Series(user_proposal, V.user_in_test.index) \
                        .reindex(self._user_list, fill_value=0).values
        item_proposal = pd.Series(item_proposal, V.item_in_test.index) \
                        .reindex(self._padded_item_list, fill_value=0).values

        V = V.reindex(self._user_list, axis=0).reindex(self._padded_item_list, axis=1)
        target_coo = V.target_csr.tocoo()

        dataset = np.transpose([
            target_coo.row, target_coo.col, k * np.ones_like(target_coo.row),
        ]).astype(int)

        G = self._extract_features(V)

        return dataset, G, user_proposal, item_proposal

    @empty_cache_on_exit
    def fit(self, *V_arr):
        dataset, G_list, user_proposal, item_proposal = zip(*[
            self._extract_task(k, V) for k, V in enumerate(V_arr)
        ])

        print("GraphConv label sizes", [len(d) for d in dataset])
        dataset = np.vstack(dataset)
        model = _GraphConv(np.array(user_proposal), np.array(item_proposal), **self._model_kw)

        N = len(dataset)
        if len(dataset) > 5:
            train_set, valid_set = random_split(dataset, [N * 4 // 5, (N - N * 4 // 5)])
        else:
            train_set = valid_set = dataset

        trainer = Trainer(
            max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])

        model.G_list = G_list
        trainer.fit(
            model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=(N > 1e4) * 4),
            DataLoader(valid_set, self.batch_size, num_workers=(N > 1e4) * 4))
        delattr(model, "G_list")

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
            self._user_list, self._padded_item_list, 'softplus')

        return S.reindex(D.user_in_test.index, axis=0).reindex(D.item_in_test.index, axis=1)
