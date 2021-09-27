import pandas as pd, numpy as np, scipy as sp
import functools, torch, gc
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from rim_experiments.util import empty_cache_on_exit, _SparseArrayWrapper, _LitValidated

def get_batch_size(shape):
    """ round to similar batch sizes """
    n_users, n_items = shape
    if torch.cuda.device_count():
        total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        total_memory = 16e9
    max_batch_size = total_memory / 8 / 10 / n_items
    n_batches = int(n_users / max_batch_size) + 1
    return int(np.ceil(n_users / n_batches))


class _LitCVX(_LitValidated):
    """ topk on v-modified scores thanks to dual decomposition """
    def __init__(self, n_users, n_items, alpha, beta, ub, v=None, lr=1):

        super().__init__()
        self.topk = np.round(alpha * n_items).astype(int)
        self.beta = beta
        self.ub = ub

        if v is None:
            v = (-1)**ub * torch.rand(n_items)
        self._v = torch.nn.Parameter(v)
        self.lr = lr


    @property
    def v(self):
        if self.ub:
            return self._v.clip(0, None) # obj - pos(v) to meet upper-constraints
        else:
            return self._v.clip(None, 0)


    @torch.no_grad()
    def forward(self, batch):
        dual_score = batch - self.v.detach().to(batch.device)[None, :]
        topk = torch.topk(dual_score, k=self.topk)
        indptr = torch.arange(len(batch) + 1).to(batch.device) * self.topk

        pi = _SparseArrayWrapper(
            indptr, topk.indices, torch.ones_like(topk.values), batch.shape)
        return pi


    def training_step(self, batch, batch_idx):
        pi = self.forward(batch)
        r = pi.mean(axis=0) - self.beta
        loss = ((self.v.detach() + r - self._v)**2).mean() / 2
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


class CVX:
    def __init__(self, score_mat, topk, C,
        constraint_type='ub', max_epochs=100, gpus=1, prefix='CVX', **kw):

        n_users, n_items = score_mat.shape
        alpha = topk / n_items
        beta = C / n_users

        self.score_max = score_mat.max()
        self.batch_size = get_batch_size(score_mat.shape)
        self.item_rec_feasible = (constraint_type=='ub' or alpha>beta)

        if not self.item_rec_feasible: # transpose
            n_users, n_items = n_items, n_users
            alpha, beta = beta, alpha

        self.model = _LitCVX(n_users, n_items, alpha, beta, constraint_type=='ub', **kw)

        tb_logger = loggers.TensorBoardLogger(
            "logs/", name=f"{prefix}-{topk}-{C}-{constraint_type}")
        self.trainer = Trainer(max_epochs=max_epochs, gpus=gpus, logger=tb_logger,
            auto_select_gpus=True, log_every_n_steps=1)
        print("trainer log at:", self.trainer.logger.log_dir)

        self.trainer.tune(self.model) # this may auto-select gpus; not sure.


    def _standardize_input(self, score_mat):
        if not self.item_rec_feasible:
            score_mat = score_mat.T
        return score_mat / self.score_max


    @empty_cache_on_exit
    def transform(self, score_mat):
        score_mat = self._standardize_input(score_mat)

        pred_batches = self.trainer.predict(self.model,
            DataLoader(score_mat, self.batch_size)
        )
        delattr(self.model, "predict_dataloader")

        pi = [pi.scipy() for pi in pred_batches]
        return sp.sparse.vstack(pi)


    @empty_cache_on_exit
    def fit(self, score_mat):
        score_mat = self._standardize_input(score_mat)

        self.trainer.fit(self.model,
            DataLoader(score_mat, self.batch_size, True),
            DataLoader(score_mat, self.batch_size), # valid on train set in offline cases
            )
        delattr(self.model, 'train_dataloader')
        delattr(self.model, 'val_dataloader')

        print("val_loss", self.model.val_loss)
        return self
