import torch, argparse, numpy as np, warnings, torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import (_LitValidated, _ReduceLRLoadCkpt, empty_cache_on_exit, extract_past_ij,
                    default_random_split, LazyScoreBase, auto_cast_lazy_score)
from .lightfm_bpr import LightFM_BPR


class _BPR_Common(_LitValidated):
    """ assumes item/user_encoder, bias_vec """
    def __init__(self, user_rec, item_rec, n_negatives, lr, weight_decay):
        super().__init__()
        self.user_rec = user_rec
        self.item_rec = item_rec
        self.n_negatives = n_negatives
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, i, j, user_kw={}):
        user_embeddings = self.user_encoder(i, **user_kw)
        item_embeddings = self.item_encoder(j)

        return (user_embeddings * item_embeddings).sum(-1) \
            + self.user_bias_vec(i, **user_kw).squeeze(-1) \
            + self.item_bias_vec(j).squeeze(-1)

    def _bpr_training_step(self, batch, user_proposal, item_proposal,
                           prior_score=None, prior_score_T=None, **kw):
        i, j = batch.T
        pos_score = self.forward(i, j, **kw)

        n_shape = (self.n_negatives, len(batch))
        loglik = []

        if self.user_rec:
            user_proposal = torch.as_tensor(user_proposal).to(batch.device)
            if prior_score_T is not None:
                ni = _mnl_w_prior(prior_score_T[j.tolist()], user_proposal, self.n_negatives)
            else:
                ni = torch.multinomial(user_proposal, np.prod(n_shape), True).reshape(n_shape)
            ni_score = self.forward(ni, j, **kw)
            loglik.append(F.logsigmoid(pos_score - ni_score))

        if self.item_rec:
            item_proposal = torch.as_tensor(item_proposal).to(batch.device)
            if prior_score is not None:
                nj = _mnl_w_prior(prior_score[i.tolist()], item_proposal, self.n_negatives)
            else:
                nj = torch.multinomial(item_proposal, np.prod(n_shape), True).reshape(n_shape)
            nj_score = self.forward(i, nj, **kw)
            loglik.append(F.logsigmoid(pos_score - nj_score))

        return -torch.stack(loglik).mean()

    def configure_optimizers(self):
        print({k: v.shape for k, v in self.named_parameters()})
        optimizer = torch.optim.Adagrad(
            self.parameters(), eps=1e-3, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = _ReduceLRLoadCkpt(
            optimizer, model=self, factor=0.25, patience=4, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": lr_scheduler, "monitor": "val_epoch_loss"
                }}


class _BPR(_BPR_Common):
    def __init__(self, n_users=None, n_items=None,
                 user_rec=True, item_rec=True, no_components=32,
                 n_negatives=10, lr=1, weight_decay=1e-5,
                 user_embeddings=None, item_embeddings=None):

        super().__init__(user_rec, item_rec, n_negatives, lr, weight_decay)

        if item_embeddings is not None or user_embeddings is not None:
            warnings.warn("setting no_components according to provided embeddings")
            no_components = item_embeddings.shape[-1] if item_embeddings is not None else \
                            user_embeddings.shape[-1]

        self.user_encoder = torch.nn.Embedding(n_users, no_components)
        self.user_bias_vec = torch.nn.Embedding(n_users, 1)
        if user_embeddings is not None:
            self.user_encoder.weight.requires_grad = False
            self.user_encoder.weight.copy_(torch.as_tensor(user_embeddings))

        self.item_encoder = torch.nn.Embedding(n_items, no_components)
        self.item_bias_vec = torch.nn.Embedding(n_items, 1)
        if item_embeddings is not None:
            self.item_encoder.weight.requires_grad = False
            self.item_encoder.weight.copy_(torch.as_tensor(item_embeddings))

    def training_step(self, batch, batch_idx):
        loss = self._bpr_training_step(batch, self.user_proposal, self.item_proposal)
        self.log("train_loss", loss, prog_bar=True)
        return loss


@torch.no_grad()
def _mnl_w_prior(S: LazyScoreBase, proposal, n_negatives):
    out = []
    for i in range(0, len(S), S.batch_size):
        batch = S[i:min(len(S), i + S.batch_size)]
        prob = (batch.eval(proposal.device) + proposal.log()).softmax(1)
        batch_out = torch.multinomial(prob, n_negatives, True)  # batch_size, n_negatives
        out.append(batch_out)
    return torch.vstack(out).T  # n_negatives, batch_size


class BPR(LightFM_BPR):
    def __init__(self, user_rec=True, item_rec=True, batch_size=10000, max_epochs=50,
                 sample_with_posterior=0.5, **kw):
        self._model_kw = {"user_rec": user_rec, "item_rec": item_rec}
        self._model_kw.update(kw)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.sample_with_posterior = sample_with_posterior
        self._transposed = False

    @empty_cache_on_exit
    def fit(self, D):
        i, j = extract_past_ij(D.user_df, D.item_df.index)
        dataset = np.transpose([i, j]).astype(int)

        N = len(dataset)
        train_set, valid_set = default_random_split(dataset)

        if 'embedding' in D.user_df:
            self._model_kw['user_embeddings'] = np.vstack(D.user_df['embedding'])
        if 'embedding' in D.item_df:
            self._model_kw['item_embeddings'] = np.vstack(D.item_df['embedding'])

        model = _BPR(len(D.user_df), len(D.item_df), **self._model_kw)
        model.user_proposal = (D.user_df['_hist_len'].values + 0.1) ** self.sample_with_posterior
        model.item_proposal = (D.item_df['_hist_len'].values + 0.1) ** self.sample_with_posterior

        trainer = Trainer(
            max_epochs=self.max_epochs, gpus=int(torch.cuda.is_available()),
            log_every_n_steps=1, callbacks=[model._checkpoint, LearningRateMonitor()])

        trainer.fit(
            model,
            DataLoader(train_set, self.batch_size, shuffle=True, num_workers=(N > 1e4) * 4),
            DataLoader(valid_set, self.batch_size, num_workers=(N > 1e4) * 4))
        model._load_best_checkpoint("best")
        for attr in ['user_proposal', 'item_proposal']:
            delattr(model, attr)

        self.D = D
        self.bpr_model = argparse.Namespace(
            user_embeddings=model.user_encoder.weight.detach().cpu().numpy(),
            item_embeddings=model.item_encoder.weight.detach().cpu().numpy(),
            user_biases=model.user_bias_vec.weight.squeeze(-1).detach().cpu().numpy(),
            item_biases=model.item_bias_vec.weight.squeeze(-1).detach().cpu().numpy(),
        )
        return self
