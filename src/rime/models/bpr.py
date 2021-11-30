import torch, argparse, numpy as np
from torch.utils.data import DataLoader, random_split
from torch.distributions.categorical import Categorical
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import _LitValidated, _ReduceLRLoadCkpt, empty_cache_on_exit, create_matrix
from .lightfm_bpr import LightFM_BPR


class _BPR(_LitValidated):
    def __init__(self, user_proposal, item_proposal,
        user_rec=True, item_rec=True, no_components=32,
        n_negatives=10, lr=1, weight_decay=1e-5,
        encode_user_ids=True):
        super().__init__()
        self.register_buffer("user_proposal", torch.as_tensor(user_proposal))
        self.register_buffer("item_proposal", torch.as_tensor(item_proposal))

        n_users = user_proposal.shape[-1]
        n_items = item_proposal.shape[-1]

        if encode_user_ids:
            self.user_encoder = torch.nn.Embedding(n_users, no_components)
            self.user_bias_vec = torch.nn.Embedding(n_users, 1)

        self.item_encoder = torch.nn.Embedding(n_items, no_components)
        self.item_bias_vec = torch.nn.Embedding(n_items, 1)
        self.log_sigmoid = torch.nn.LogSigmoid()

        self.user_rec = user_rec
        self.item_rec = item_rec
        self.n_negatives = n_negatives
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, i, j):
        return (self.user_encoder(i) * self.item_encoder(j)).sum(-1) \
            + self.user_bias_vec(i).squeeze(-1) + self.item_bias_vec(j).squeeze(-1)

    def _bpr_training_step(self, batch, user_proposal, item_proposal, **kw):
        i, j = batch.T
        pos_score = self.forward(i, j, **kw)

        n_shape = (self.n_negatives, len(batch))
        loglik = []

        if self.user_rec:
            ni = torch.multinomial(user_proposal, np.prod(n_shape), True).reshape(n_shape)
            ni_score = self.forward(ni, j, **kw)
            loglik.append(self.log_sigmoid(pos_score - ni_score))

        if self.item_rec:
            nj = torch.multinomial(item_proposal, np.prod(n_shape), True).reshape(n_shape)
            nj_score = self.forward(i, nj, **kw)
            loglik.append(self.log_sigmoid(pos_score - nj_score))

        return -torch.stack(loglik).mean()

    def training_step(self, batch, batch_idx):
        loss = self._bpr_training_step(batch, self.user_proposal, self.item_proposal)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(),
            eps=1e-3, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = _ReduceLRLoadCkpt(optimizer, model=self,
            factor=0.25, patience=4, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": lr_scheduler, "monitor": "val_epoch_loss"
            }}


class BPR(LightFM_BPR):
    def __init__(self, user_rec=True, item_rec=True, batch_size=10000, max_epochs=50, **kw):
        self._model_kw = {"user_rec": user_rec, "item_rec": item_rec}
        self._model_kw.update(kw)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._transposed=False

    @empty_cache_on_exit
    def fit(self, D):
        ij_target = create_matrix(D.event_df, D.user_df.index, D.item_df.index, 'ij')
        dataset = np.array(ij_target, dtype=int).T

        N = len(dataset)
        if len(dataset) > 5:
            train_set, valid_set = random_split(dataset, [N*4//5, (N - N*4//5)])
        else:
            train_set = valid_set = dataset

        user_proposal = (D.user_df['_hist_len'].values + 0.1) ** 0.5
        item_proposal = (D.item_df['_hist_len'].values + 0.1) ** 0.5

        model = _BPR(user_proposal, item_proposal, **self._model_kw)

        trainer = Trainer(max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])
        trainer.fit(model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=(N>1e4)*4),
            DataLoader(valid_set, self.batch_size, num_workers=(N>1e4)*4))

        best_model_path = model._checkpoint.best_model_path
        best_model_score = model._checkpoint.best_model_score
        if best_model_score is not None:
            print(f"done fit; best checkpoint {best_model_path} with score {best_model_score}")

        self.D = D
        self.bpr_model = argparse.Namespace(
            user_embeddings=model.user_encoder.weight.detach().cpu().numpy(),
            item_embeddings=model.item_encoder.weight.detach().cpu().numpy(),
            user_biases=model.user_bias_vec.weight.squeeze(-1).detach().cpu().numpy(),
            item_biases=model.item_bias_vec.weight.squeeze(-1).detach().cpu().numpy(),
            )
        return self
