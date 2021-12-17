import torch, functools, numpy as np
from .third_party.word_language_model.model import TransformerModel
from .rnn import (RNN, Trainer, _LitRNNModel, _LitValidated, _collate_fn, LearningRateMonitor)


class _LitTransformerModel(_LitRNNModel, _LitValidated):
    def __init__(self, ntoken, *args, truncated_bptt_steps=None, lr=0.1 / 4, **kw):
        super(_LitValidated, self).__init__()
        self.model = TransformerModel(ntoken, *args, **kw)
        self.loss = torch.nn.NLLLoss(ignore_index=0)
        self.ntoken = ntoken
        self.lr = lr

    def training_step(self, batch, batch_idx):
        """ max length defined through truncated_input_steps=256 """
        x, y = batch[0].T, batch[1].T   # transpose to TN layout
        out = self.model(x, True)
        # print(batch_idx, out.softmax(dim=-1).detach().cpu().numpy().round(2))
        loss = self.loss(out.view(-1, self.ntoken), y.view(-1))
        self.log("train_loss", loss)
        return loss

    def forward(self, batch):
        """ output user embedding at lengths-1 positions """
        TN_inp, lengths = batch
        mask = self.model._generate_square_subsequent_mask(len(TN_inp)).to(TN_inp.device)

        TNC_enc = self.model.encoder(TN_inp) * np.sqrt(self.model.ninp)
        TNC_enc = self.model.pos_encoder(TNC_enc)
        TNC_out = self.model.transformer_encoder(TNC_enc, mask)
        return self._decode_last(TNC_out, lengths)


class Transformer(RNN):
    def __init__(
        self, item_df, max_item_size=int(30e3),
        num_hidden=128, nlayers=2, max_epochs=20, nhead=2, lr=0.1 / 4,
        gpus=int(torch.cuda.is_available()),
        truncated_input_steps=256, batch_size=64,
        load_from_checkpoint=None
    ):

        sorted_item_df = item_df['_hist_len'].sort_values(ascending=False)
        self._padded_item_list = [None] + sorted_item_df.index.tolist()[:max_item_size]
        self._truncated_input_steps = truncated_input_steps
        self._collate_fn = functools.partial(
            _collate_fn,
            tokenize={k: i for i, k in enumerate(self._padded_item_list)},
            truncated_input_steps=truncated_input_steps)

        self.model = _LitTransformerModel(
            len(self._padded_item_list),
            num_hidden, nhead, num_hidden, nlayers, 0, lr=lr)

        if load_from_checkpoint is not None:
            self.model.load_state_dict(
                torch.load(load_from_checkpoint)['state_dict'])

        self.trainer = Trainer(
            max_epochs=max_epochs, gpus=gpus,
            callbacks=[self.model._checkpoint, LearningRateMonitor()])
        print("trainer log at:", self.trainer.logger.log_dir)
        self.batch_size = batch_size
