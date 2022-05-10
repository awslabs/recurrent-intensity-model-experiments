from transformers import AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM
import torch, pandas as pd, numpy as np, functools, json, warnings
from rime.util import (matrix_reindex, LazyDenseMatrix, _to_cuda, empty_cache_on_exit,
                       explode_user_titles)
from tqdm import tqdm


class ItemKNN:
    """ softplus(x'y / temperature + log_p_y) for numerical stability """

    def __init__(self, item_df, batch_size=100,
                 item_pop_power=1, item_pop_pseudo=0.01,
                 model_name='bert-base-uncased',  # gpt2
                 pooling=None,  # cls or mean
                 temperature=None, gamma=0.5):

        assert "TITLE" in item_df or "embedding" in item_df, "require TITLE or embedding"

        self.item_index = item_df.index
        self.item_biases = item_pop_power * np.log(item_df['_hist_len'].values + item_pop_pseudo)
        self.batch_size = batch_size
        if temperature is None:
            temperature = {
                'bert-base-uncased': 10,
                'gpt2': 100,
            }[model_name]
        if pooling is None:
            pooling = 'cls' if 'bert' in model_name else 'mean'
        self.pooling = pooling
        self.temperature = temperature
        self.gamma = gamma

        if "embedding" in item_df:
            self.item_embeddings = np.vstack(item_df["embedding"].values)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            assert self.tokenizer.padding_side == 'right', "expect right padding"
            if model_name == 'gpt2':
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if model_name.startswith('bert'):
                self.model = BertForMaskedLM.from_pretrained(model_name)
            else:  # gpt2
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()  # eval mode
            self.item_embeddings = self._compute_embeddings(item_df["TITLE"].values)

    @empty_cache_on_exit
    @torch.no_grad()
    def _compute_embeddings(self, titles):
        """ find embedding of a batch of sequences """
        with _to_cuda(self.model) as model:
            embeddings = []

            for batch in tqdm(np.split(titles, range(0, len(titles), self.batch_size)[1:])):
                inputs = self.tokenizer(batch.tolist(), padding=True, return_tensors='pt')
                if hasattr(self.model, 'bert'):
                    for key in inputs.keys():  # 'input_ids', 'attention_mask', 'token_type_ids'
                        inputs[key] = inputs[key][:, :512]
                    offset = 1  # [cls] seq [sep]
                    hidden_states = self.model.bert(**inputs.to(model.device))[0]
                else:
                    offset = 0
                    hidden_states = self.model.transformer(**inputs.to(model.device))[0]

                if self.pooling == 'mean':
                    segments = [slice(offset, n - offset) for n in
                                inputs['attention_mask'].sum(1).tolist()]
                elif self.pooling == 'cls':
                    segments = [slice(0, 1) for _ in inputs['attention_mask']]
                else:
                    raise NotImplementedError

                hidden_states = torch.vstack([  # mean-pooling on causal lm states
                    x[slc].mean(0, keepdims=True) for x, slc in zip(hidden_states, segments)])

                embeddings.append(hidden_states.double().cpu().numpy())

        return np.vstack(embeddings)

    @empty_cache_on_exit
    def transform(self, D):
        explode_embeddings, splits, weights = explode_user_titles(
            D.user_in_test['_hist_items'],
            pd.Series(self.item_embeddings.tolist(), self.item_index),
            self.gamma, pad_title=np.zeros_like(self.item_embeddings[0]).tolist())

        user_embeddings = np.vstack([w @ x for w, x in zip(
            np.split(weights, splits), np.split(np.vstack(explode_embeddings), splits)
        )]) / self.temperature  # not very useful due to linearity in softplus

        item_reindex = lambda x, fill_value=0: matrix_reindex(
            x, self.item_index, D.item_in_test.index, axis=0, fill_value=fill_value)
        return (LazyDenseMatrix(user_embeddings) @ item_reindex(self.item_embeddings).T
                + item_reindex(self.item_biases, fill_value=-np.inf)).softplus()
