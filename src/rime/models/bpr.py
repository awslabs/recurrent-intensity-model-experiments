import torch, argparse, numpy as np
from torch.utils.data import DataLoader, random_split
from torch.distributions.categorical import Categorical
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import _LitValidated, _ReduceLRLoadCkpt, empty_cache_on_exit
from .lightfm_bpr import LightFM_BPR


class _BPR(_LitValidated):
    def __init__(self, user_prop, item_prop, no_components,
        n_negatives=10, lr=1, weight_decay=1e-5):
        super().__init__()
        self.register_buffer("user_prop", torch.as_tensor(user_prop))
        self.register_buffer("item_prop", torch.as_tensor(item_prop))

        self.user_encoder = torch.nn.Embedding(len(user_prop), no_components)
        self.item_encoder = torch.nn.Embedding(len(item_prop), no_components)
        self.user_bias_vec = torch.nn.Embedding(len(user_prop), 1)
        self.item_bias_vec = torch.nn.Embedding(len(item_prop), 1)
        self.log_sigmoid = torch.nn.LogSigmoid()

        self.n_negatives = n_negatives
        self.lr = lr
        self.weight_decay = weight_decay

    def _bilinear_score(self, i, j):
        return (self.user_encoder(i) * self.item_encoder(j)).sum(-1) \
            + self.user_bias_vec(i).squeeze(-1) + self.item_bias_vec(j).squeeze(-1)

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


class BPR(LightFM_BPR):
    def __init__(self, no_components=32, batch_size=10000, max_epochs=50):
        self.__dict__.update(**locals())
        self._transposed=False

    @empty_cache_on_exit
    def fit(self, D):
        user2ind = {k:i for i,k in enumerate(D.user_df.index)}
        item2ind = {k:i for i,k in enumerate(D.item_df.index)}
        dataset = np.array([
            [user2ind[k] for k in D.event_df['USER_ID']],
            [item2ind[k] for k in D.event_df['ITEM_ID']],
            ]).T

        N = len(dataset)
        if len(dataset) > 5:
            train_set, valid_set = random_split(dataset, [N*4//5, (N - N*4//5)])
            num_workers = 4
        else:
            train_set = valid_set = dataset
            num_workers = 0

        user_prop = (D.user_df['_hist_len'].values + 0.1) ** 0.5
        item_prop = (D.item_df['_hist_len'].values + 0.1) ** 0.5

        model = _BPR(user_prop, item_prop, self.no_components)

        trainer = Trainer(max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])
        trainer.fit(model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(valid_set, self.batch_size, num_workers=num_workers))

        best_model_path = model._checkpoint.best_model_path
        best_model_score = model._checkpoint.best_model_score
        if best_model_score is not None:
            print(f"done fit; best checkpoint {best_model_path} with score {best_model_score}")

        self.D = D
        self.bpr_model = argparse.Namespace(
            user_embeddings=model.user_encoder.weight.detach().cpu().numpy(),
            item_embeddings=model.item_encoder.weight.detach().cpu().numpy(),
            user_biases=model.user_bias_vec.weight.detach().cpu().numpy().ravel(),
            item_biases=model.item_bias_vec.weight.detach().cpu().numpy().ravel(),
            )
        return self
