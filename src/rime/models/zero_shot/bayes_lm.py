from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import torch, pandas as pd, numpy as np, warnings, functools
from rime.util import matrix_reindex, _to_cuda, empty_cache_on_exit, extract_last_titles
from tqdm import tqdm


class BayesLM:
    """ p(y|x) propto lm(x|y) * p(y), where x is the last item in user history
    and y is a candidate item.

    https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto
    """

    def __init__(self, item_df, max_num_candidates=None, batch_size=100,
                 prompt="a user will watch {y} after watching {x}",
                 item_pop_power=1, item_pop_pseudo=0.01, temperature=1,
                 candidate_selection_method=None, model_name='gpt2', text_column_name='TITLE'):

        assert text_column_name in item_df, f"require {text_column_name} as data(y)"

        self.item_df = item_df.copy()
        self.item_df['log_p_y'] = item_pop_power * np.log(item_df['_hist_len'] + item_pop_pseudo)

        if max_num_candidates is None:
            warnings.warn("please set max_num_candidates, default=2 only for testing purposes")
            max_num_candidates = 2

        self.max_num_candidates = max_num_candidates
        self.batch_size = batch_size
        self.prompt = prompt
        self.temperature = temperature
        self.text_column_name = text_column_name
 
        if candidate_selection_method is None:
            candidate_selection_method = 'greedy' if item_pop_power > 0 else 'sample'
        self.candidate_selection_method = candidate_selection_method

        # huggingface model initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = 'right'
        if model_name == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # eval mode

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    @torch.no_grad()
    def _log_p_x_given_y(self, Y, x, device):
        """ evaluate p_x_given_y for each y in Y; label = x.expand(...) """

        batch_size = len(Y)
        sequences = [self.prompt.format(y=y, x=x) for y in Y]
        inputs = self.tokenizer(sequences, padding=True, return_tensors='pt').to(device)
        labels = self.tokenizer(x, return_tensors='pt')['input_ids'].to(device)
        labels = torch.vstack([labels for _ in range(batch_size)])

        seq_len = inputs['attention_mask'].sum(1).tolist()
        target_len = labels.shape[1]

        if hasattr(self.model, "lm_head"):  # 20% faster
            hidden_states = self.model.transformer(**inputs)[0]
            hidden_states = torch.vstack([x[n - target_len - 1: n - 1]
                                          for x, n in zip(hidden_states, seq_len)])
            logits = self.model.lm_head(hidden_states)
        else:
            logits = self.model(**inputs).logits
            logits = torch.vstack([x[n - target_len - 1: n - 1]
                                   for x, n in zip(logits, seq_len)])

        loss = self.loss(logits, labels.reshape(-1))
        return (-loss).reshape(labels.shape).mean(1).tolist()

    @functools.lru_cache(1)
    @empty_cache_on_exit
    def transform(self, D):
        """ generate score matrix by evaluating top or random items in test """

        user_last_titles = extract_last_titles(D.user_in_test['_hist_items'],
                                               self.item_df[self.text_column_name])

        sorted_items = self.item_df[self.item_df.index.isin(D.item_in_test.index)] \
                           .sort_values('_hist_len', ascending=False, kind='mergesort')
        p_y = torch.as_tensor(sorted_items['log_p_y'].values).softmax(0).numpy()
        num_candidates = int(min(self.max_num_candidates, len(sorted_items)))

        with _to_cuda(self.model) as model:
            scores = []

            for x in tqdm(user_last_titles.values):
                if self.candidate_selection_method == 'greedy':
                    ind = np.arange(num_candidates)
                else:
                    ind = np.random.choice(len(sorted_items), num_candidates, False, p_y)

                candidate_titles = sorted_items[self.text_column_name].values[ind]
                log_p_y = sorted_items['log_p_y'].values[ind]

                log_p_x_given_y = np.hstack([
                    self._log_p_x_given_y(Y, x, model.device) for Y in
                    np.split(candidate_titles, range(0, num_candidates, self.batch_size)[1:])
                ]) / self.temperature
                p_y_given_x = torch.as_tensor(log_p_x_given_y + log_p_y).softmax(0).numpy()

                this = matrix_reindex(p_y_given_x[None, :], sorted_items.index.values[ind],
                                      D.item_in_test.index, axis=1, fill_value=0)
                scores.append(this)

        return np.vstack(scores)  # dense matrix with shape = user_in_test x item_in_test
