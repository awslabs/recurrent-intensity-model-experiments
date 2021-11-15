import pandas as pd, numpy as np, scipy as sp
import torch, itertools, os, warnings
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from ..util import empty_cache_on_exit, _LitValidated, get_batch_size, score_op
from ..util.cvx_bisect import dual_solve_u, dual_clip, primal_solution


class CVX:
    def __init__(self, score_mat, topk, C, constraint_type='ub', device='cpu',
        max_epochs=None, min_epsilon=1e-10, gpus=int(torch.cuda.is_available()),
        prefix='CVX'):

        n_users, n_items = score_mat.shape
        alpha = topk / n_items
        beta = C / n_users

        if hasattr(score_mat, "collate_fn"):
            # assert not score_mat.has_nan(), "score matrix has nan"
            self.score_max = float(score_op(score_mat, "max", device))
            self.score_min = float(score_op(score_mat, "min", device))
        else:
            self.score_max = float(score_mat.max())
            self.score_min = float(score_mat.min())

        print(f"entering {prefix} CVX score (min={self.score_min}, max={self.score_max})")
        self.device = device

        if max_epochs is None:
            if 0<np.min(alpha)<=np.max(alpha)<1 and 0<np.min(beta)<=np.max(beta)<1:
                max_epochs = 100
            else: # trivial cases with u or v being 0
                max_epochs = 10

        self._model_args = (
            n_users, n_items, alpha, beta, constraint_type, 0.1 / max(score_mat.shape),
            max_epochs, min_epsilon)

        tb_logger = loggers.TensorBoardLogger("logs/",
            name=f"{prefix}-{topk}-{np.mean(C):.1f}-{constraint_type}-{not np.isscalar(C)}")

        self._trainer_kw = dict(max_epochs=max_epochs, gpus=gpus, logger=tb_logger,
            log_every_n_steps=1,
            callbacks=[ModelCheckpoint()], # change default save path from . to logger path
            )


    @empty_cache_on_exit
    def transform(self, score_mat):
        score_mat = score_mat * (1./self.score_max)
        batch_size = self.model.batch_size

        def fn(i):
            batch = score_mat[i:min(i+batch_size, len(score_mat))]
            return self.model.forward(batch, device=self.device)

        pi = np.vstack([fn(i) for i in range(0, len(score_mat), batch_size)])
        return pi


    @empty_cache_on_exit
    def fit(self, score_mat):
        score_mat = score_mat * (1./self.score_max)

        model = _LitCVX(*self._model_args)
        trainer = Trainer(**self._trainer_kw)
        print("trainer log at:", trainer.logger.log_dir)

        collate_fn = getattr(score_mat[0], "collate_fn",
            torch.utils.data.dataloader.default_collate)

        trainer.fit(model,
            DataLoader(score_mat, model.batch_size, True, collate_fn=collate_fn),
            )
        v = model.v.detach().cpu().numpy()
        print('v', pd.Series(v.ravel()).describe().to_dict())

        self.model = _LitCVX(*self._model_args,
            v=model.v, epsilon=model.epsilon)
        return self


class _LitCVX(LightningModule):
    def __init__(self, n_users, n_items, alpha, beta, constraint_type, gtol,
        max_epochs=100, min_epsilon=1e-10, v=None, epsilon=1):

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.constraint_type = constraint_type
        self.gtol = gtol

        self.batch_size = get_batch_size((n_users, n_items)) #, frac=0.05)
        n_batches = n_users / self.batch_size
        self.lr = n_items / n_batches

        self.epsilon = epsilon
        self.epsilon_gamma = (min_epsilon / epsilon) ** (1/max_epochs)

        if v is None:
            if constraint_type == 'ub':
                v = torch.rand(n_items)
            elif constraint_type == 'lb':
                v = -torch.rand(n_items)
            else: # eq
                v = torch.rand(n_items) * 2 - 1
        self.v = torch.nn.Parameter(v)


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        self.epsilon *= self.epsilon_gamma
        self.log("epsilon", self.epsilon, prog_bar=True)

    @torch.no_grad()
    def forward(self, batch, v=None, epsilon=None, device="cpu"):
        if hasattr(batch, "eval"):
            batch = batch.eval(device).detach()
        else:
            batch = torch.as_tensor(batch)

        if v is None:
            v = self.v.detach().to(batch.device)

        if epsilon is None:
            epsilon = self.epsilon

        u, _ = dual_solve_u(v, batch, self.alpha, epsilon, gtol=self.gtol)
        u = dual_clip(u, "ub")
        pi = primal_solution(u, v, batch, epsilon)
        return pi.cpu().numpy()


    def training_step(self, batch, batch_idx):
        if hasattr(batch, "eval"):
            batch = batch.eval(self.device).detach()
        else:
            batch = torch.as_tensor(batch)

        u, u_iters = dual_solve_u(
            self.v.detach(), batch, self.alpha, self.epsilon, gtol=self.gtol)
        u = dual_clip(u, "ub")
        self.log("u_iters", u_iters, prog_bar=True)

        v, v_iters = dual_solve_u(u, batch.T, self.beta, self.epsilon, gtol=self.gtol)
        v = dual_clip(v, self.constraint_type)
        self.log("v_iters", v_iters, prog_bar=True)

        loss = ((self.v - v)**2).mean() / 2
        self.log("train_loss", loss)
        return loss
