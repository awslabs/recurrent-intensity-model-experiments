import pandas as pd, numpy as np, scipy as sp
import torch, itertools, os, warnings
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from rim_experiments.util import empty_cache_on_exit, _LitValidated, \
                            get_batch_size, get_best_gpus
from rim_experiments.util.cvx_bisect import dual_solve_u


class CVX:
    def __init__(self, score_mat, topk, C, constraint_type='ub', device='cpu',
        max_epochs=100, min_epsilon=1e-10,
        gpus=get_best_gpus() if torch.cuda.is_available() else 0,
        prefix='CVX'):

        n_users, n_items = score_mat.shape
        alpha = topk / n_items
        beta = C / n_users

        if hasattr(score_mat, "gpu_max"):
            assert not score_mat.has_nan(), "score matrix has nan"
            self.score_max = float(score_mat.gpu_max(device=device))
            self.score_min = float(score_mat.gpu_min(device=device))
        else:
            self.score_max = float(score_mat.max())
            self.score_min = float(score_mat.min())

        print(f"entering {prefix} CVX score in ({self.score_min}, {self.score_max})")

        self._model_args = (
            n_users, n_items, alpha, beta, constraint_type,
            max_epochs, min_epsilon)

        tb_logger = loggers.TensorBoardLogger(
            "logs/", name=f"{prefix}-{topk}-{C}-{constraint_type}")

        self._trainer_kw = dict(max_epochs=max_epochs, gpus=gpus, logger=tb_logger,
            log_every_n_steps=1)


    @empty_cache_on_exit
    def transform(self, score_mat):
        cost_mat = score_mat * (-1./self.score_max)
        batch_size = self.model.batch_size

        def fn(i):
            batch = cost_mat[i:min(i+batch_size, len(cost_mat))]
            return self.model.forward(batch)

        pi = np.vstack([fn(i) for i in range(0, len(cost_mat), batch_size)])
        return pi


    @empty_cache_on_exit
    def fit(self, score_mat):
        cost_mat = score_mat * (-1./self.score_max)

        model = _LitCVX(*self._model_args)
        trainer = Trainer(**self._trainer_kw)
        print("trainer log at:", trainer.logger.log_dir)

        collate_fn = getattr(cost_mat[0], "collate_fn",
            torch.utils.data.dataloader.default_collate)

        trainer.fit(model,
            DataLoader(cost_mat, model.batch_size, True, collate_fn=collate_fn),
            )
        print("train_loss", model.train_loss)

        self.model = _LitCVX(*self._model_args,
            v=model.v, epsilon=model.epsilon)
        return self


class _LitCVX(LightningModule):
    def __init__(self, n_users, n_items, alpha, beta, constraint_type,
        max_epochs=100, min_epsilon=1e-10, v=None, epsilon=1):

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.constraint_type = constraint_type

        if torch.cuda.device_count():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_batch_size = total_memory / 8 / 10 / n_items
        else:
            max_batch_size = float("inf")
        n_batches = int(n_users / max_batch_size) + 1

        self.lr = n_items / n_batches
        self.batch_size = int(np.ceil(n_users / n_batches))

        self.epsilon = epsilon
        self.epsilon_gamma = min_epsilon ** (1/max_epochs)

        if v is None:
            if constraint_type == 'ub':
                v = -torch.rand(n_items)
            elif constraint_type == 'lb':
                v = torch.rand(n_items)
            else: # eq
                v = torch.rand(n_items) * 2 - 1
        self.v = torch.nn.Parameter(v)


    @staticmethod
    def _clip(v, constraint_type):
        if constraint_type == 'ub':
            return v.clip(None, 0)
        elif constraint_type == 'lb':
            return v.clip(0, None)
        else: # eq
            return v


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def on_epoch_start(self):
        self.epsilon *= self.epsilon_gamma
        self.log("epsilon", self.epsilon, prog_bar=True)

    @torch.no_grad()
    def forward(self, batch, v=None, epsilon=None):
        if hasattr(batch, "eval"):
            batch = batch.eval(self.device).detach()
        else:
            batch = torch.as_tensor(batch)

        if v is None:
            v = self.v.detach().to(batch.device)

        if epsilon is None:
            epsilon = self.epsilon

        u, _ = _solve(v[None, :] - batch, self.alpha, epsilon)
        u = u.clip(None, 0)
        z = (-batch + u[:, None] + v[None, :]) / epsilon
        # print(get_grad(z, self.alpha))
        return torch.sigmoid(z).cpu().numpy()


    def training_step(self, batch, batch_idx):
        if hasattr(batch, "eval"):
            batch = batch.eval(self.device).detach()
        else:
            batch = torch.as_tensor(batch)

        u, _ = _solve(self.v[None, :] - batch, self.alpha, self.epsilon)
        u = u.clip(None, 0)
        v, _ = _solve(u[None, :] - batch.T, self.beta, self.epsilon)
        v = self._clip(v, self.constraint_type)
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
    topk = min(int(alpha*add.shape[1]) + 1, add.shape[1])
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


if int(os.environ.get('CVX_BISECT', 1)):
    print("CVX_BISECT")

    @torch.no_grad()
    def _solve(add, alpha, epsilon, return_negative_u=True):
        """ find u s.t. E_y[pi(x,y)] == alpha,
        where pi(x,y) = sigmoid((add(x,y) - u(y)) / epsilon)
        """
        assert np.isscalar(alpha), "only supports scalar for simplicity"

        if alpha < 0 or alpha > 1:
            warnings.warn(f"clipping alpha={alpha} to [0, 1]")
            alpha = np.clip(alpha, 0, 1)

        alpha = torch.as_tensor(alpha).to(add.device)
        epsilon = torch.as_tensor(epsilon).to(add.device)

        _primal = lambda u: torch.sigmoid((add - u[:, None]) / epsilon)
        _grad_u = lambda u: alpha - _primal(u).mean(axis=1) # monotone with u

        z = alpha.log() - (1-alpha).log()

        if alpha == 0 or alpha == 1: # z is +- infinity
            u = -z * torch.ones_like(add[:, 0])
            return (-u if return_negative_u else u, (u-u).max())

        u_min = add.amin(axis=1) - z * epsilon - 1e-3
        u_max = add.amax(axis=1) - z * epsilon + 1e-3

        assert (_grad_u(u_min) <= 0).all()
        assert (_grad_u(u_max) >= 0).all()

        for i in range(50):
            u = (u_min + u_max) / 2
            g = _grad_u(u)
            u_min = torch.where(g<0, u, u_min)
            u_max = torch.where(g>0, u, u_max)

        return (-u if return_negative_u else u, (u_max-u_min).max())
