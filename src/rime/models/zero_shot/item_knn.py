from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import torch, pandas as pd, numpy as np
from rime.util import (create_second_order_dataframe, _to_cuda, empty_cache_on_exit,
                       extract_last_titles)
from tqdm import tqdm


class ItemKNN:
    """ softplus(x'y / temperature + log_p_y) for numerical stability """

    def __init__(self, item_df, batch_size=100,
                 item_pop_power=1, item_pop_pseudo=0.01,
                 temperature=1000, model='gpt2', text_column_name='TITLE'):

        assert text_column_name in item_df, f"require {text_column_name} as data(y)"

        self.item_df = item_df.copy()
        self.item_df['log_p_y'] = item_pop_power * np.log(item_df['_hist_len'] + item_pop_pseudo)
        self.batch_size = batch_size
        self.temperature = temperature
        self.text_column_name = text_column_name

        # huggingface model initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.eval()  # eval mode

        self.item_embeddings = self._compute_embeddings(
            self.item_df[self.text_column_name].values)

    @torch.no_grad()
    def _compute_embeddings(self, titles):
        """ find embedding of a batch of sequences """
        embeddings = []
        with _to_cuda(self.model) as model:
            for batch in np.split(titles, range(0, len(titles), self.batch_size)[1:]):
                inputs = self.tokenizer(batch.tolist(), padding=True, return_tensors='pt')
                seq_len = inputs['attention_mask'].sum(1).tolist()
                hidden_states = self.model.transformer(**inputs.to(model.device))[0]

                for x, n in zip(hidden_states, seq_len):
                    embeddings.append(x[n - 1].tolist())

        return np.vstack(embeddings)

    @empty_cache_on_exit
    def transform(self, D):
        user_last_titles = extract_last_titles(D.user_in_test['_hist_items'],
                                               self.item_df[self.text_column_name])
        user_embeddings = self._compute_embeddings(user_last_titles.values) / self.temperature

        return create_second_order_dataframe(
            user_embeddings, self.item_embeddings, None, self.item_df['log_p_y'].values,
            D.user_in_test.index, self.item_df.index.values, 'softplus'
        ).reindex(D.item_in_test.index, axis=1, fill_value=0)
