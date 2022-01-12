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

        self.item_index = item_df.index
        self.batch_size = batch_size
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

        self.item_embeddings = self._compute_embeddings(item_df[text_column_name].values)
        self.item_biases = item_pop_power * np.log(item_df['_hist_len'].values + item_pop_pseudo)

    @torch.no_grad()
    def _compute_embeddings(self, titles):
        """ find embedding of a batch of sequences """
        with _to_cuda(self.model) as model:
            embeddings = []

            for batch in tqdm(np.split(titles, range(0, len(titles), self.batch_size)[1:])):
                inputs = self.tokenizer(batch.tolist(), padding=True, return_tensors='pt')
                if hasattr(self.model, 'bert'):
                    inputs['input_ids'] = inputs['input_ids'][:, :512]
                    inputs['attention_mask'] = inputs['attention_mask'][:, :512]
                    inputs['token_type_ids'] = inputs['token_type_ids'][:, :512]

                    hidden_states = self.model.bert(**inputs.to(model.device))[0]
                    hidden_states = hidden_states[:, 0]  # cls token at beginning of sequence
                else:  # gpt2 does not work well
                    seq_len = inputs['attention_mask'].sum(1).tolist()

                    hidden_states = self.model.transformer(**inputs.to(model.device))[0]
                    hidden_states = torch.vstack([  # mean-pooling on causal lm states
                        x[:n].mean(0, keepdims=True) for x, n in zip(hidden_states, seq_len)])

                embeddings.append(hidden_states.double().cpu().numpy())

        return np.vstack(embeddings)

    @empty_cache_on_exit
    def transform(self, D):
        explode_embeddings, splits, weights = explode_user_titles(  # directly use embeddings
            D.user_in_test['_hist_items'],
            pd.Series(self.item_embeddings.tolist(), self.item_index),
            self.gamma, pad_title=np.zeros_like(self.item_embeddings[0]).tolist())

        user_embeddings = np.vstack([w @ x for w, x in zip(
            np.split(weights, splits), np.split(np.vstack(explode_embeddings), splits)
        )]) / self.temperature

        return create_second_order_dataframe(
            user_embeddings, self.item_embeddings, None, self.item_biases,
            D.user_in_test.index, self.item_index, 'softplus'
        ).reindex(D.item_in_test.index, axis=1, fill_value=0)
