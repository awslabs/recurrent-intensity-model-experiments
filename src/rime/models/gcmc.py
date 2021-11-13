import torch, argparse, numpy as np, dgl
from torch.utils.data import DataLoader, random_split
from torch.distributions.categorical import Categorical
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import (_LitValidated, _ReduceLRLoadCkpt,
    empty_cache_on_exit, create_second_order_dataframe)
from .bpr import BPR
import dgl.function as fn


class _GCMC(_LitValidated):
    """ module to compute user RFM embedding. register G and user_recency before calling """
    def __init__(self, user_prop, item_prop, no_components,
        n_negatives=10, lr=0.1, weight_decay=1e-5, recency_boundaries=[0.1, 0.3, 1, 3, 10]):
        super().__init__()
        self.register_buffer("user_prop", torch.as_tensor(user_prop))
        self.register_buffer("item_prop", torch.as_tensor(item_prop))
        self.register_buffer("recency_boundaries", torch.as_tensor(recency_boundaries))

        self.item_encoder = torch.nn.Embedding(len(item_prop), no_components)
        self.item_bias_vec = torch.nn.Embedding(len(item_prop), 1)
        self.conv = dgl.nn.pytorch.conv.GraphConv(no_components, no_components, "none")
        self.recency_encoder = torch.nn.Embedding(len(recency_boundaries)+1, 1)
        self.log_sigmoid = torch.nn.LogSigmoid()

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
        user_recency = G.nodes['user'].data['recency'].to(i.device)

        recency_buckets = torch.bucketize(user_recency, self.recency_boundaries)
        return self.recency_encoder(recency_buckets)[i]

    def _bilinear_score(self, i, j):
        score = (self.user_encoder(i) * self.item_encoder(j)).sum(-1)
        score = score + self.user_bias_vec(i).squeeze(-1)
        return score + self.item_bias_vec(j).squeeze(-1)

    def training_step(self, batch, batch_idx):
        i, j = batch.T
        n_shape = (self.n_negatives, len(batch))
        ni = torch.multinomial(self.user_prop, np.prod(n_shape), True).reshape(n_shape)
        nj = torch.multinomial(self.item_prop, np.prod(n_shape), True).reshape(n_shape)

        pos_score = self._bilinear_score(i, j)
        ni_score = self._bilinear_score(ni, j)
        nj_score = self._bilinear_score(i, nj)

        loglik = torch.stack([self.log_sigmoid(pos_score - ni_score),
                              self.log_sigmoid(pos_score - nj_score)])
        loss = -loglik.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(),
            eps=1e-3, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = _ReduceLRLoadCkpt(optimizer, model=self,
            factor=0.25, patience=4, min_lr=self.lr*1e-3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": lr_scheduler, "monitor": "val_epoch_loss"
            }}


class GCMC:
    def __init__(self, item_df, no_components=32, batch_size=10000, max_epochs=50):
        """ item_df = D.training_data.item_df """
        self.__dict__.update(**locals())
        self._padded_item_list = [None] + item_df.index.tolist()

    def _extract_labels(self, D):
        D = D.reindex(self._padded_item_list, axis=1)
        target_coo = D.target_csr.tocoo()
        i = [i for i,d in zip(target_coo.row, target_coo.data) for k in range(int(d))]
        j = [j for j,d in zip(target_coo.col, target_coo.data) for k in range(int(d))]

        user_prop = (D.user_in_test['_hist_len'].values + 0.1) ** 0.5
        item_prop = (D.item_in_test['_hist_len'].values + 0.1) ** 0.5
        return (i, j), user_prop, item_prop

    def _extract_features(self, D):
        D = D.reindex(self._padded_item_list, axis=1)
        user2ind = {k:i for i,k in enumerate(D.user_in_test.index)}
        item2ind = {k:i for i,k in enumerate(D.item_in_test.index)}

        event_df = D.training_data.event_df[
            D.training_data.event_df['USER_ID'].isin(user2ind) &
            D.training_data.event_df['ITEM_ID'].isin(item2ind)
        ]
        i = [user2ind[k] for k in event_df['USER_ID']]
        j = [item2ind[k] for k in event_df['ITEM_ID']]

        n_users = len(D.user_in_test)
        i = np.hstack([i, np.arange(n_users)])
        j = np.hstack([j, np.zeros_like(j, shape=n_users)]) # padding items

        user_recency = D.user_in_test['_timestamps'].apply(
            lambda x: x[-1]-x[-2] if len(x)>1 else np.inf
            )
        return (i, j), user_recency

    @empty_cache_on_exit
    def fit(self, V):
        ij_target, user_prop, item_prop = self._extract_labels(V)

        dataset = np.array(ij_target, dtype=int).T
        N = len(dataset)
        if len(dataset) > 5:
            train_set, valid_set = random_split(dataset, [N*4//5, (N - N*4//5)])
        else:
            train_set = valid_set = dataset

        model = _GCMC(user_prop, item_prop, self.no_components)
        trainer = Trainer(max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])

        ij_source, user_recency = self._extract_features(V)
        G = dgl.heterograph(
            {('user','source','item'): ij_source,},
            {"user": len(user_recency), "item": len(self._padded_item_list)}
            ).reverse()
        G.nodes['user'].data['recency'] = torch.as_tensor(user_recency.values)

        model.G = G
        trainer.fit(model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=(N>5)*4),
            DataLoader(valid_set, self.batch_size, num_workers=(N>5)*4))
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
        ij_source, user_recency = self._extract_features(D)

        G = dgl.heterograph(
            {('user','source','item'): ij_source,},
            {"user": len(user_recency), "item": len(self._padded_item_list)}
            ).reverse()
        G.nodes['user'].data['recency'] = torch.as_tensor(user_recency.values)

        i = torch.arange(G.num_nodes('user'))
        user_embeddings = self.model.user_encoder(i, G).detach().cpu().numpy()
        user_biases = self.model.user_bias_vec(i, G).detach().cpu().numpy().ravel()

        S = create_second_order_dataframe(
            user_embeddings, self.item_embeddings, user_biases, self.item_biases,
            user_recency.index, self._padded_item_list, 'sigmoid')

        return S.reindex(D.item_in_test.index, axis=1)
