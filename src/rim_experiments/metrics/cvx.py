import pandas as pd, numpy as np, scipy as sp
import functools, torch, gc
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from rim_experiments.util import empty_cache_on_exit, _LitValidated, get_batch_size


class CVX:
    def __init__(self, score_mat, topk, C, constraint_type='ub',
        max_epochs=100, min_epsilon=1e-10, gpus=1, prefix='CVX'):

        n_users, n_items = score_mat.shape
        alpha = topk / n_items
        beta = C / n_users

        self.score_max = score_mat.max()
        if constraint_type == 'lb':
            assert alpha >= beta, "requires item_rec feasible in online cases"

        self.model = _LitCVX(
            n_users, n_items, alpha, beta, constraint_type=='ub',
            max_epochs, min_epsilon)

        tb_logger = loggers.TensorBoardLogger(
            "logs/", name=f"{prefix}-{topk}-{C}-{constraint_type}")

        self.trainer = Trainer(max_epochs=max_epochs, gpus=gpus, logger=tb_logger,
            auto_select_gpus=True, log_every_n_steps=1)
        print("trainer log at:", self.trainer.logger.log_dir)

    @empty_cache_on_exit
    def transform(self, score_mat):
        cost_mat = -score_mat / self.score_max

        pi = np.vstack(
            self.trainer.predict(self.model,
                DataLoader(cost_mat, self.model.batch_size))
            )
        delattr(self.model, "predict_dataloader")
        return pi


    @empty_cache_on_exit
    def fit(self, score_mat):
        cost_mat = -score_mat / self.score_max

        self.trainer.tune(self.model) # auto_select_gpus
        self.trainer.fit(self.model,
            DataLoader(cost_mat, self.model.batch_size, True),
            )
        print("train_loss", self.model.train_loss)

        delattr(self.model, "train_dataloader")
        delattr(self.trainer, "train_dataloader")
        return self


class _LitCVX(LightningModule):
    def __init__(self, n_users, n_items, alpha, beta, user_rec_ub,
        max_epochs=100, min_epsilon=1e-10, v=None, epsilon=1):

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.user_rec_ub = user_rec_ub

        self.batch_size = get_batch_size((n_users, n_items))
        self.lr = n_items / (n_users / self.batch_size)

        self.epsilon = epsilon
        self.epsilon_gamma = min_epsilon ** (1/max_epochs)

        if v is None:
            v = (-1)**user_rec_ub * torch.rand(n_items)
        self.v = torch.nn.Parameter(v)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def on_epoch_start(self):
        self.epsilon *= self.epsilon_gamma
        self.log("epsilon", self.epsilon, prog_bar=True)

    @torch.no_grad()
    def forward(self, batch):
        v = self.v.detach().to(batch.device)
        u, _ = _solve(v[None, :] - batch, self.alpha, self.epsilon)
        u = u.clip(None, 0)
        z = (-batch + u[:, None] + v[None, :]) / self.epsilon
        # print(get_grad(z, self.alpha))
        return torch.sigmoid(z).cpu().numpy()

    def training_step(self, batch, batch_idx):
        u, _ = _solve(self.v[None, :] - batch, self.alpha, self.epsilon)
        u = u.clip(None, 0)
        v, _ = _solve(u[None, :] - batch.T, self.beta, self.epsilon)
        v = v.clip(None, 0) if self.user_rec_ub else v.clip(0, None) # lb
        loss = ((self.v - v)**2).mean() / 2
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.train_loss = torch.stack([o['loss'] for o in outputs]).mean()


def get_fmin(z, u, epsilon, alpha):
    return epsilon*torch.logaddexp(torch.zeros_like(z), z).mean(axis=1) - u*alpha

def get_grad(z, alpha):
    return torch.sigmoid(z).mean(axis=1) - alpha

def get_hess(z, epsilon):
    return (torch.sigmoid(z) * torch.sigmoid(-z)).mean(axis=1) / epsilon

@torch.no_grad()
def _solve(add, alpha, epsilon, n_iters=10, n_bt=4, tol=1e-5):
    """ minimize epsilon*log(1+exp((u+add) / epsilon)).mean() - u*alpha
    whose gradient is sigmoid((u+add)/epsilon).mean() = alpha
    """
    topk = int(alpha*add.shape[1]) + 1
    u = -torch.topk(add, topk, sorted=False).values.amin(1)

    for i in range(n_iters):
        u0 = u
        z = (u[:, None] + add)/epsilon
        fmin = get_fmin(z, u, epsilon, alpha)
        grad = get_grad(z, alpha)
        if grad.abs().mean().tolist() < tol:
            break
        hess = get_hess(z, epsilon)
        eta = torch.ones_like(u)

        for bt in range(n_bt):
            u = u0 - eta * grad / hess.clip(1e-3, None)
            z = (u[:, None] + add)/epsilon
            fnew = get_fmin(z, u, epsilon, alpha)
            success = (fnew - fmin <= 0.5 * grad * (u - u0) + 1e-6)

            if success.all():
                break
            else:
                eta = torch.where(success, eta, eta*0.5)

    assert not torch.isnan(u).any(), "nan in solve"
    return u, grad.abs().mean().tolist()
