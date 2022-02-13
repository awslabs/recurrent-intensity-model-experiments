import torch, functools, numpy as np
from .third_party.word_language_model import TransformerModel
from .rnn import (RNN, Trainer, _LitRNNModel, _LitValidated, _collate_fn, LearningRateMonitor,
                  get_top_items)


class _LitTransformerModel(_LitRNNModel, _LitValidated):
    def __init__(self, ntoken, *args, truncated_bptt_steps=None, lr=0.1 / 4, **kw):
        super(_LitValidated, self).__init__()
        self.model = TransformerModel(ntoken, *args, **kw)
        self.loss = torch.nn.NLLLoss(ignore_index=0)
        self.ntoken = ntoken
        self.lr = lr

    def training_step(self, batch, batch_idx):
        """ max length defined through truncated_input_steps=256 """
        x, y = batch[0], batch[1]   # TN layout
        out = self.model(x, True)
        # print(batch_idx, out.softmax(dim=-1).detach().cpu().numpy().round(2))
        loss = self.loss(out.view(-1, self.ntoken), y.view(-1))
        self.log("train_loss", loss)
        return loss


class Transformer(RNN):
    def __init__(
        self, item_df, max_item_size=int(30e3),
        num_hidden=128, nlayers=2, max_epochs=20, nhead=2, lr=0.1 / 4,
        gpus=int(torch.cuda.is_available()),
        truncated_input_steps=256, batch_size=64,
        load_from_checkpoint=None, tie_weights=True,
    ):

        self._padded_item_list = [None] + get_top_items(item_df, max_item_size).index.tolist()
        self._tokenize = {k: i for i, k in enumerate(self._padded_item_list)}
        self._truncated_input_steps = truncated_input_steps

        self.model = _LitTransformerModel(
            len(self._padded_item_list),
            num_hidden, nhead, num_hidden, nlayers, 0, lr=lr, tie_weights=tie_weights)

        if load_from_checkpoint is not None:
            self.model.load_state_dict(
                torch.load(load_from_checkpoint)['state_dict'])

        self.trainer = Trainer(
            max_epochs=max_epochs, gpus=gpus,
            callbacks=[self.model._checkpoint, LearningRateMonitor()])
        print("trainer log at:", self.trainer.logger.log_dir)
        self.batch_size = batch_size
