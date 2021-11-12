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
    def __init__(self, user_prior, item_prior, no_components,
        n_negatives=10, lr=1, weight_decay=1e-5):
        super().__init__()
        self.register_buffer("user_prior", torch.as_tensor(user_prior))
        self.register_buffer("item_prior", torch.as_tensor(item_prior))

        self.item_encoder = torch.nn.Embedding(len(item_prior), no_components)
        self.item_bias_vec = torch.nn.Embedding(len(item_prior), 1)
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.conv = dgl.nn.pytorch.conv.GraphConv(no_components, no_components)

        self.n_negatives = n_negatives
        self.lr = lr
        self.weight_decay = weight_decay

        # self.G = G        # set G for training and inference separately
        # self.G.nodes['item'].data['weight'] = self.item_encoder.weight

    def user_encoder(self, i, G=None):
        if G is None:
            G = self.G
        out = self.conv(G.to(i.device), self.item_encoder.weight)
        # self.G.update_all()
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        # block, *_ = sampler.sample_blocks(self.G, {'user': i.ravel()})
        # out = conv(block, block.nodes['item_src'].data['weight'])
        # out = out.view((*i.shape, -1))
        return out[i]

    def _bilinear_score(self, i, j):
        score = (self.user_encoder(i) * self.item_encoder(j)).sum(-1)
        return score + self.item_bias_vec(j).squeeze(-1)

    def training_step(self, batch, batch_idx):
        i, j = batch.T
        n_shape = (self.n_negatives, len(batch))
        ni = torch.multinomial(self.user_prior, np.prod(n_shape), True).reshape(n_shape)
        nj = torch.multinomial(self.item_prior, np.prod(n_shape), True).reshape(n_shape)

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

        user_prior = (D.user_in_test['_hist_len'].values + 0.1) ** 0.5
        item_prior = (D.item_in_test['_hist_len'].values + 0.1) ** 0.5
        return (i, j), user_prior, item_prior

    def _extract_features(self, D):
        D = D.reindex(self._padded_item_list, axis=1)
        user2ind = {k:i for i,k in enumerate(D.user_in_test.index)}
        item2ind = {k:i for i,k in enumerate(D.item_in_test.index)}

        event_df = D.training_data.event_df[
            D.training_data.event_df['USER_ID'].isin(user2ind) &
            D.training_data.event_df['ITEM_ID'].isin(item2ind)
        ].join(
            D.user_in_test['_timestamps']
                .apply(lambda x: x[-1]).to_frame("TEST_START_TIME"),
            on='USER_ID'
            )
        i = [user2ind[k] for k in event_df['USER_ID']]
        j = [item2ind[k] for k in event_df['ITEM_ID']]
        dt = (event_df['TEST_START_TIME'] - event_df['TIMESTAMP']).values / D.horizon

        n_users = len(D.user_in_test)
        i = np.hstack([i, np.arange(n_users)])
        j = np.hstack([j, np.zeros_like(j, shape=n_users)]) # padding items
        dt = np.hstack([dt, np.ones_like(dt, shape=n_users) * np.inf])
        return (i, j), dt, D.user_in_test.index

    @empty_cache_on_exit
    def fit(self, V):
        ij_target, user_prior, item_prior = self._extract_labels(V)
        ij_source, dt, user_index = self._extract_features(V)

        G = dgl.heterograph(
            {('user','source','item'): ij_source,},
            {"user": len(user_index), "item": len(self._padded_item_list)}
            ).reverse()
        G.edata['dt'] = torch.as_tensor(dt)
        G.update_all(lambda x: None, fn.min("dt", "min_dt"))

        model = _GCMC(user_prior, item_prior, self.no_components)

        dataset = np.array(ij_target, dtype=int).T
        N = len(dataset)
        if len(dataset) > 5:
            train_set, valid_set = random_split(dataset, [N*4//5, (N - N*4//5)])
            num_workers = 4
        else:
            train_set = valid_set = dataset
            num_workers = 0

        trainer = Trainer(max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])

        model.G = G
        trainer.fit(model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(valid_set, self.batch_size, num_workers=num_workers))

        best_model_path = model._checkpoint.best_model_path
        best_model_score = model._checkpoint.best_model_score
        if best_model_score is not None:
            print(f"done fit; best checkpoint {best_model_path} with score {best_model_score}")

        self.item_index = self._padded_item_list
        self.item_embeddings = model.item_encoder.weight.detach().cpu().numpy()
        self.item_biases = model.item_bias_vec.weight.detach().cpu().numpy().ravel()
        self.model = model
        delattr(model, "G")

        return self

    def transform(self, D):
        ij_source, dt, user_index = self._extract_features(D)

        G = dgl.heterograph(
            {('user','source','item'): ij_source,},
            {"user": len(user_index), "item": len(self._padded_item_list)}
            ).reverse()
        G.edata['dt'] = torch.as_tensor(dt)
        G.update_all(lambda x: None, fn.min("dt", "min_dt"))

        i = torch.arange(G.num_nodes('user'))
        user_embeddings = self.model.user_encoder(i, G).detach().cpu().numpy()

        S = create_second_order_dataframe(
            user_embeddings, self.item_embeddings, None, self.item_biases,
            user_index, self._padded_item_list, 'sigmoid')

        return S.reindex(D.item_in_test.index, axis=1)
