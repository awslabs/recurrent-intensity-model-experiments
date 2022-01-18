import numpy as np
import functools, warnings

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from .third_party.word_language_model.model import RNNModel

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from ..util import (_LitValidated, empty_cache_on_exit, LowRankDataFrame, _ReduceLRLoadCkpt,
                    default_random_split, get_top_items)


class RNN:
    def __init__(
        self, item_df, max_item_size=int(30e3),
        num_hidden=128, nlayers=2, max_epochs=20, gpus=int(torch.cuda.is_available()),
        truncated_input_steps=256, truncated_bptt_steps=32, batch_size=64,
        load_from_checkpoint=None
    ):
        self._padded_item_list = [None] + get_top_items(item_df, max_item_size).index.tolist()
        self._truncated_input_steps = truncated_input_steps
        self._collate_fn = functools.partial(
            _collate_fn,
            tokenize={k: i for i, k in enumerate(self._padded_item_list)},
            truncated_input_steps=truncated_input_steps)

        self.model = _LitRNNModel(
            'GRU', len(self._padded_item_list),
            num_hidden, num_hidden, nlayers, 0, True,
            truncated_bptt_steps=truncated_bptt_steps)

        if load_from_checkpoint is not None:
            self.model.load_state_dict(
                torch.load(load_from_checkpoint)['state_dict'])

        self.trainer = Trainer(
            max_epochs=max_epochs, gpus=gpus,
            callbacks=[self.model._checkpoint, LearningRateMonitor()])
        print("trainer log at:", self.trainer.logger.log_dir)
        self.batch_size = batch_size

    @functools.lru_cache(2)
    @empty_cache_on_exit
    @torch.no_grad()
    def transform(self, D):
        dataset = D.user_in_test['_hist_items'].values
        collate_fn = functools.partial(self._collate_fn, training=False)
        m, n_events, sample_y = _get_dataset_stats(dataset, collate_fn)
        print(f"transforming {m} users with {n_events} events, "
              f"truncated@{self._truncated_input_steps} per user")
        print(f"sample_y={sample_y}")

        batches = self.trainer.predict(
            self.model,
            dataloaders=DataLoader(dataset.tolist(), 1000, collate_fn=collate_fn))

        user_hidden, user_log_bias = [np.concatenate(x) for x in zip(*batches)]
        ind_logits = np.hstack([
            user_hidden, user_log_bias[:, None], np.ones_like(user_log_bias)[:, None]
        ])

        item_hidden = self.model.model.decoder.weight.detach().cpu().numpy()
        item_log_bias = self.model.model.decoder.bias.detach().cpu().numpy()
        col_logits = np.hstack([
            item_hidden, np.ones_like(item_log_bias)[:, None], item_log_bias[:, None]
        ])

        return LowRankDataFrame(
            ind_logits, col_logits, D.user_in_test.index,
            self._padded_item_list, act='exp'
        ).reindex(D.item_in_test.index, axis=1, fill_value=0)

    @empty_cache_on_exit
    def fit(self, D):
        dataset = D.user_df[D.user_df['_hist_len'] > 0]['_hist_items'].values
        collate_fn = functools.partial(self._collate_fn, training=True)
        m, n_events, sample_y = _get_dataset_stats(dataset, collate_fn)
        print(f"fitting {m} users with {n_events} events, "
              f"truncated@{self._truncated_input_steps} per user")
        print(f"sample_y={sample_y}")

        train_set, valid_set = default_random_split(dataset)
        self.trainer.fit(
            self.model,
            DataLoader(train_set, self.batch_size, collate_fn=collate_fn, shuffle=True),
            DataLoader(valid_set, self.batch_size, collate_fn=collate_fn),)
        self.model._load_best_checkpoint("best")

        for name, param in self.model.named_parameters():
            print(name, param.data.shape)
        return self


def _collate_fn(batch, tokenize, truncated_input_steps, training):
    if truncated_input_steps > 0:
        batch = [seq[-truncated_input_steps:] for seq in batch]
    batch = [[0] + [tokenize[x] for x in seq if x in tokenize] for seq in batch]
    batch = [torch.tensor(seq, dtype=torch.int64) for seq in batch]
    batch, lengths = pad_packed_sequence(pack_sequence(batch, False))
    if training:
        return (batch[:-1].T, batch[1:].T)  # TBPTT assumes NT layout
    else:
        return (batch, lengths)  # RNN default TN layout


def _get_dataset_stats(dataset, collate_fn):
    truncated_input_steps = collate_fn.keywords['truncated_input_steps']
    n_events = sum([min(truncated_input_steps, len(x)) for x in dataset])
    sample = next(iter(DataLoader(dataset, 2, collate_fn=collate_fn, shuffle=True)))
    return len(dataset), n_events, sample[1]


class _LitRNNModel(_LitValidated):
    def __init__(self, *args, truncated_bptt_steps, lr=0.1, **kw):
        super().__init__()
        self.model = RNNModel(*args, **kw)
        self.loss = torch.nn.NLLLoss(ignore_index=0)
        self.truncated_bptt_steps = truncated_bptt_steps
        self.lr = lr

    def forward(self, batch):
        """ output user embedding at lengths-1 positions """
        TN_layout, lengths = batch
        hiddens = self.model.init_hidden(len(lengths))
        TNC_out, _ = self.model.rnn(self.model.encoder(TN_layout), hiddens)
        return self._decode_last(TNC_out, lengths)

    def _decode_last(self, TNC_out, lengths):
        last_hidden = TNC_out[lengths - 1, np.arange(len(lengths))]
        pred_logits = self.model.decoder(last_hidden)
        log_bias = -pred_logits.logsumexp(1)
        return last_hidden.cpu().numpy(), log_bias.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), eps=1e-3, lr=self.lr)
        lr_scheduler = _ReduceLRLoadCkpt(
            optimizer, model=self, factor=0.25, patience=4, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": lr_scheduler, "monitor": "val_epoch_loss"
                }}

    def training_step(self, batch, batch_idx, hiddens=None):
        """ truncated_bptt_steps pass batch[:][:, slice] and hiddens """
        x, y = batch[0].T, batch[1].T   # transpose to TN layout
        if hiddens is None:
            hiddens = self.model.init_hidden(x.shape[1])
        else:
            hiddens = hiddens.detach()
        out, hiddens = self.model(x, hiddens)
        loss = self.loss(out, y.view(-1))
        self.log("train_loss", loss)
        return {'loss': loss, 'hiddens': hiddens}
