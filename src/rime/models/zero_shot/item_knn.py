from transformers import AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM
import torch, pandas as pd, numpy as np, functools
from rime.util import (create_second_order_dataframe, _to_cuda, empty_cache_on_exit,
                       explode_user_titles)
from tqdm import tqdm


class ItemKNN:
    """ softplus(x'y / temperature + log_p_y) for numerical stability """

    def __init__(self, item_df, batch_size=100,
                 item_pop_power=1, item_pop_pseudo=0.01,
                 model_name='bert-base-uncased',  # gpt2
                 temperature=None, gamma=0.5, text_column_name='TITLE'):

        assert text_column_name in item_df, f"require {text_column_name} as data(y)"

        self.item_titles = item_df[text_column_name]
        self.batch_size = batch_size
        item_biases = item_pop_power * np.log(item_df['_hist_len'] + item_pop_pseudo)

        if temperature is None:
            temperature = {
                'bert-base-uncased': 10,
                'gpt2': 100,
            }[model_name]
        self.temperature = temperature
        self.gamma = gamma

        # huggingface model initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert self.tokenizer.padding_side == 'right', "expect right padding"
        if model_name == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if model_name.startswith('bert'):
            self.model = BertForMaskedLM.from_pretrained(model_name)
        else:  # gpt2
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # eval mode

        self.item_embeddings = self._compute_embeddings(self.item_titles.values)
        self.item_biases = item_biases.values.astype(self.item_embeddings.dtype)

    @torch.no_grad()
    def _compute_embeddings(self, titles):
        """ find embedding of a batch of sequences """
        with _to_cuda(self.model) as model:
            embeddings = []

            for batch in np.split(titles, range(0, len(titles), self.batch_size)[1:]):
                inputs = self.tokenizer(batch.tolist(), padding=True, return_tensors='pt')
                seq_len = inputs['attention_mask'].sum(1).tolist()
                if hasattr(self.model, 'bert'):
                    hidden_states = self.model.bert(**inputs.to(model.device))[0]
                    hidden_states = hidden_states[:, 0]  # cls token at beginning of sequence
                else:  # gpt2 does not work well
                    hidden_states = self.model.transformer(**inputs.to(model.device))[0]
                    hidden_states = torch.vstack([  # mean-pooling on causal lm states
                        x[:n].mean(0, keepdims=True) for x, n in zip(hidden_states, seq_len)])

                embeddings.append(hidden_states)

        return torch.vstack(embeddings).cpu().numpy().astype(float)

    @functools.lru_cache(2)
    @empty_cache_on_exit
    def transform(self, D):
        explode_titles, splits, weights = explode_user_titles(
            D.user_in_test['_hist_items'], self.item_titles, self.gamma)

        explode_embeddings = self._compute_embeddings(explode_titles.values)

        user_embeddings = np.vstack([w @ x for w, x in zip(
            np.split(weights, splits), np.split(explode_embeddings, splits)
        )]) / self.temperature

        return create_second_order_dataframe(
            user_embeddings, self.item_embeddings, None, self.item_biases,
            D.user_in_test.index, self.item_titles.index, 'softplus'
        ).reindex(D.item_in_test.index, axis=1, fill_value=0)
