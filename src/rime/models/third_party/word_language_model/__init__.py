import numpy as np
from .model import RNNModel as _RNNModel, TransformerModel as _TransformerModel


class _ForwardLastPrediction:
    def forward_last_prediction(self, input_time_by_batch, lengths):
        all_hidden, *_ = self.forward_all_hidden(input_time_by_batch)
        hidden = all_hidden[lengths - 1, np.arange(len(lengths))]
        logsumexp = self.decoder(hidden).logsumexp(1)
        return hidden, -logsumexp


class RNNModel(_RNNModel, _ForwardLastPrediction):
    def forward_all_hidden(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape[1])
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output, hidden


class TransformerModel(_TransformerModel, _ForwardLastPrediction):
    def __init__(self, *args, tie_weights=False, **kwargs):
        super().__init__(*args, **kwargs)
        if tie_weights:
            sizes = (self.encoder.weight.shape, self.decoder.weight.shape)
            assert sizes[0] == sizes[1], f"tie weights size conflict {sizes}"
            self.decoder.weight = self.encoder.weight

    def forward_all_hidden(self, src, has_mask="true and ignored"):
        mask = self._generate_square_subsequent_mask(len(src)).to(src.device)

        src = self.encoder(src) * np.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        return (output,)
